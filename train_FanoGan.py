import os
import torch.utils.data as TUD
from torchvision.utils import save_image
import torch.autograd as autograd
from tqdm import tqdm
from torch import nn
from data.save_arch import *
from util.Gen_ITS2CLR_util import *
import torch.optim as optim
from data.format_data import *
from data.pseudo_labels import *
from data.bag_loader import *
from data.instance_loader import *
from loss.FocalLoss import *
from util.eval_util import *
from config import *
import matplotlib.pyplot as plt
import seaborn as sns
torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



def compute_gradient_penalty(D, real_samples, fake_samples, device):
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones((real_samples.size(0), 1), requires_grad=False, device=device)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def compute_diversity_loss(generated_images):
    # Compute pairwise distances between generated samples
    diff = generated_images.unsqueeze(1) - generated_images.unsqueeze(0)
    diff = diff.view(diff.size(0), diff.size(1), -1)
    distances = torch.norm(diff, dim=2)
    
    # We want to maximize these distances
    diversity_loss = -torch.mean(distances)
    return diversity_loss


def train_wgangp(generator, discriminator, dataloader, num_epochs=10, latent_dim=100):
    # Even smaller learning rates
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0, 0.9))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0, 0.9))

    generator.to(device)
    discriminator.to(device)

    lambda_gp = 10
    drift_penalty = 0.001 # keep loss values close to 0
    diversity_weight = 0.1
    
    # Add dynamic n_gen adjustment
    def adjust_n_gen(d_loss, g_loss):
        gap = d_loss - g_loss # Positive gap means D is behind
        
        n_gen = 1
        if gap > 2:
            n_gen = 1
        elif gap > 1:
            n_gen = 2
        elif gap < -2:
            n_gen = 4
        elif gap < -1:
            n_gen = 3
            
        return n_gen

    for epoch in range(num_epochs):
        for i, (real_imgs, instance_labels, unique_id) in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            z = torch.randn(batch_size, latent_dim, device=device)
            fake_imgs = generator(z).detach()
            
            
            real_validity = discriminator(real_imgs)
            fake_validity = discriminator(fake_imgs)
            
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs, fake_imgs, device)
            
            drift = (real_validity ** 2 + fake_validity ** 2).mean()
            d_loss = (-torch.mean(real_validity) + torch.mean(fake_validity) + 
                     lambda_gp * gradient_penalty + drift_penalty * drift)

            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            current_n_gen = adjust_n_gen(d_loss.item(), g_loss.item() if 'g_loss' in locals() else 0)
            for _ in range(current_n_gen):
                z = torch.randn(batch_size, latent_dim, device=device)
                optimizer_G.zero_grad()

                gen_imgs = generator(z)
                fake_validity = discriminator(gen_imgs)
                
                g_drift = fake_validity ** 2
                g_loss = -torch.mean(fake_validity) + drift_penalty * g_drift.mean() + diversity_weight * compute_diversity_loss(gen_imgs)

                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                optimizer_G.step()

            if i % 100 == 0:
                print(
                    f"[Epoch {epoch}/{num_epochs}] "
                    f"[Batch {i}/{len(dataloader)}] "
                    f"[D loss: {d_loss.item():.4f}] "
                    f"[G loss: {g_loss.item():.4f}]"
                )

            if i % 400 == 0:
                save_dir = "results/images"
                os.makedirs(save_dir, exist_ok=True)
                save_image(gen_imgs.data[:25], f"{save_dir}/{epoch}_{i}.png", 
                          nrow=5, normalize=True)
    # Save models
    torch.save(generator.state_dict(), "results/generator")
    torch.save(discriminator.state_dict(), "results/discriminator")
    
    
def train_encoder_izif(generator, discriminator, encoder, 
                       dataloader, device, num_epochs=10, kappa=.0):
    generator.load_state_dict(torch.load("results/generator"))
    discriminator.load_state_dict(torch.load("results/discriminator"))

    generator.to(device).eval()
    discriminator.to(device).eval()
    encoder.to(device)

    criterion = nn.MSELoss()

    optimizer_E = torch.optim.Adam(encoder.parameters(), lr=0.0001, betas=(0, 0.9))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_E, gamma=0.95)

    os.makedirs("results/images_e", exist_ok=True)

    padding_epoch = len(str(num_epochs))
    padding_i = len(str(len(dataloader)))

    batches_done = 0
    for epoch in range(num_epochs):
        for i, (imgs, instance_labels, unique_id) in enumerate(dataloader):

            # Configure input
            real_imgs = imgs.to(device)

            # ----------------
            #  Train Encoder
            # ----------------

            optimizer_E.zero_grad()

            # Generate a batch of latent variables
            z = encoder(real_imgs)

            # Generate a batch of images
            fake_imgs = generator(z)

            # Real features
            real_features = discriminator.forward_features(real_imgs)
            # Fake features
            fake_features = discriminator.forward_features(fake_imgs)

            # izif architecture
            loss_imgs = criterion(fake_imgs, real_imgs)
            loss_features = criterion(fake_features, real_features)
            e_loss = loss_imgs + kappa * loss_features

            e_loss.backward()
            #torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer_E.step()
            

            # Output training log every n_critic steps
            if i % 100 == 0:
                print(f"[Epoch {epoch:{padding_epoch}}/{num_epochs}] "
                      f"[Batch {i:{padding_i}}/{len(dataloader)}] "
                      f"[E loss: {e_loss.item():3f}]")

            if i % 400 == 0:                
                # Save original images
                save_image(real_imgs[:25],
                            f"results/images_e/{epoch}_{i}_original.png",
                            nrow=5, normalize=True)

                # Save reconstructed images
                save_image(fake_imgs.data[:25],
                            f"results/images_e/{epoch}_{i}_reconstructed.png",
                            nrow=5, normalize=True)
                                
        scheduler.step()

    torch.save(encoder.state_dict(), "results/encoder")
    
    
    
def test_anomaly_detection(generator, discriminator, encoder,
                           dataloader_train, prefix, device, kappa=1.0):
    generator.load_state_dict(torch.load("results/generator"))
    discriminator.load_state_dict(torch.load("results/discriminator"))
    encoder.load_state_dict(torch.load("results/encoder"))

    generator.to(device).eval()
    discriminator.to(device).eval()
    encoder.to(device).eval()

    criterion = nn.MSELoss(reduction='none')
    anomaly_scores = {'0': [], '1': [], '-1': []}
    img_distances = {'0': [], '1': [], '-1': []}
    z_distances = {'0': [], '1': [], '-1': []}
    
    for (imgs, instance_labels, unique_id) in tqdm(dataloader_train):
        real_img = imgs.to(device)
        real_z = encoder(real_img)
        fake_img = generator(real_z)
        fake_z = encoder(fake_img)

        real_feature = discriminator.forward_features(real_img)
        fake_feature = discriminator.forward_features(fake_img)

        img_distance = criterion(fake_img, real_img).mean(dim=(1,2,3))
        loss_feature = criterion(fake_feature, real_feature).mean(dim=(1,2,3))
        anomaly_score = img_distance + kappa * loss_feature
        z_distance = criterion(fake_z, real_z).mean(dim=1)

        for i in range(len(instance_labels)):
            label = str(instance_labels[i].item())
            anomaly_scores[label].append(anomaly_score[i].item())
            img_distances[label].append(img_distance[i].item()) 
            z_distances[label].append(z_distance[i].item())

    with open(f"results/metrics_{prefix}.txt", "w") as f:
        for label in ['0', '1', '-1']:
            f.write(f"\n{'Benign' if label=='0' else 'Labeled Positive' if label=='1' else 'Positive Bags'} Images:\n")
            f.write(f"Image Distance - Mean: {np.mean(img_distances[label]):.4f}, Std: {np.std(img_distances[label]):.4f}\n")
            f.write(f"Anomaly Score - Mean: {np.mean(anomaly_scores[label]):.4f}, Std: {np.std(anomaly_scores[label]):.4f}\n")
            f.write(f"Z Distance - Mean: {np.mean(z_distances[label]):.4f}, Std: {np.std(z_distances[label]):.4f}\n")
        
        # Overall stats
        f.write("\nOverall Statistics:\n")
        all_scores = anomaly_scores['0'] + anomaly_scores['1'] + anomaly_scores['-1']
        f.write(f"Total Mean Anomaly Score: {np.mean(all_scores):.4f}\n")
        f.write(f"Min Score: {min(all_scores):.4f}\n")
        f.write(f"Max Score: {max(all_scores):.4f}\n")
        
    # Create distribution plots
    plt.figure(figsize=(15, 5))

    # Anomaly Score Distribution
    plt.subplot(1, 3, 1)
    sns.kdeplot(data=anomaly_scores['0'], label='Benign')
    sns.kdeplot(data=anomaly_scores['1'], label='Labeled Positive')
    sns.kdeplot(data=anomaly_scores['-1'], label='Positive Bags')
    plt.title('Anomaly Score Distribution')
    plt.xlabel('Score')
    plt.legend()

    # Image Distance Distribution 
    plt.subplot(1, 3, 2)
    sns.kdeplot(data=img_distances['0'], label='Benign')
    sns.kdeplot(data=img_distances['1'], label='Labeled Positive')
    sns.kdeplot(data=img_distances['-1'], label='Positive Bags')
    plt.title('Image Distance Distribution')
    plt.xlabel('Distance')
    plt.legend()

    # Z Distance Distribution
    plt.subplot(1, 3, 3)
    sns.kdeplot(data=z_distances['0'], label='Benign')
    sns.kdeplot(data=z_distances['1'], label='Labeled Positive')
    sns.kdeplot(data=z_distances['-1'], label='Positive Bags')
    plt.title('Z Distance Distribution')
    plt.xlabel('Distance')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'results/distributions_{prefix}.png')
    plt.close()




if __name__ == '__main__':

    # Config
    model_version = '1'
    head_name = "FanoGan_Test2"
    data_config = LesionDataConfig  # or LesionDataConfig
    
    config = build_config(model_version, head_name, data_config)
    bags_train, bags_val = prepare_all_data(config)
    num_classes = len(config['label_columns']) + 1
    num_labels = len(config['label_columns'])

    # Create Model
    from archs.model_FanoGan import *
    class Opt:
        def __init__(self):
            # Model architecture parameters
            self.channels = 1  # Number of channels in the image
            self.img_size = config['img_size']  # Size of images (assumes square)
            self.latent_dim = 128  # Size of z latent vector

    opt = Opt()
    
    """transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])"""
        
    transform = T.Compose([
        T.ToTensor(),
        T.Grayscale(num_output_channels=1),
        T.Normalize(mean=[0.449], std=[0.226])
    ])

    generator = Generator(opt)
    discriminator = Discriminator(opt)
    encoder = Encoder(opt)

    # Used the instance predictions from bag training to update the Instance Dataloader
    instance_dataset_train = Instance_Dataset(bags_train, [], transform=transform, only_negative = True)
    instance_dataloader_train = TUD.DataLoader(instance_dataset_train, batch_size=config['instance_batch_size'], num_workers=4, collate_fn = collate_instance, pin_memory=True, persistent_workers=True)
    
    # Phase 1: Train WGAN-GP
    #train_wgangp(generator, discriminator, instance_dataloader_train, latent_dim = opt.latent_dim)

    # Phase 2: Train Encoder
    train_encoder_izif(generator, discriminator, encoder, instance_dataloader_train, device)
    
    
    instance_dataset_train = Instance_Dataset(bags_train, [], transform=transform, only_negative = False, warmup=False)
    instance_dataset_val = Instance_Dataset(bags_val, [], transform=transform, only_negative = False, warmup=False)
    instance_dataloader_train = TUD.DataLoader(instance_dataset_train, batch_size=config['instance_batch_size'], num_workers=4, collate_fn = collate_instance, pin_memory=True, persistent_workers=True)
    instance_dataloader_val = TUD.DataLoader(instance_dataset_val, batch_size=config['instance_batch_size'], collate_fn = collate_instance)
    
    test_anomaly_detection(generator, discriminator, encoder, instance_dataloader_train, 'train', device)
    test_anomaly_detection(generator, discriminator, encoder, instance_dataloader_val, 'val', device)