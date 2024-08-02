import argparse, os, cv2, time, datetime, sys
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils.datasets import *
import hrnet
from utils.cGAN import *
from copy import deepcopy
from utils.util import SSI_Loss

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device_ids = [0]

cv2.setNumThreads(0)
#cv2.ocl.setUseOpenCL(False)

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--batch_size", type=int, default=2, help="size of the batch")
parser.add_argument("--meta_batch_size", type=int, default=5, help="size of the meta-learning mini-batch")
parser.add_argument("--outer_step_size", type=float, default=0.1)
parser.add_argument("--ML_mode", type=int, default=1, help="0: None, 1: GT with noise")
parser.add_argument("--n_epochs", type=int, default=401, help="number of epochs of training")
parser.add_argument("--n_FT_epochs", type=int, default=401, help="number of finetune epochs of training")
parser.add_argument("--img_size", type=int, default=512, help="resize the image size")
parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between model checkpoints")
parser.add_argument("--train_data_path", type=str, default="", help="path of data for training")
parser.add_argument("--FT_data_path", type=str, default="", help="path of real data for fine-tuning")
parser.add_argument("--model_save_path", type=str, default="", help="path of saved model")

opt = parser.parse_args()

# cuda
cuda = True if torch.cuda.is_available() else False

# Network
generator = hrnet.__dict__["hrnetv2"]()
discriminator = Discriminator()

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_size // 2 ** 4, opt.img_size // 2 ** 4)

# loss
criterion_MSE = torch.nn.MSELoss()
criterion_L1 = torch.nn.L1Loss()
criterion_SSI = SSI_Loss()
if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_L1.cuda()
    criterion_MSE.cuda()
    criterion_SSI.cuda()

    generator = nn.DataParallel(generator, device_ids=device_ids)
    discriminator = nn.DataParallel(discriminator, device_ids=device_ids)
    criterion_L1 = nn.DataParallel(criterion_L1, device_ids=device_ids)
    criterion_MSE = nn.DataParallel(criterion_MSE, device_ids=device_ids)
    criterion_SSI = nn.DataParallel(criterion_SSI, device_ids=device_ids)

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# optimizer
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_OA = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# data transforms
transforms_ = [
    transforms.Resize((opt.img_size, opt.img_size), transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
]

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / (_range + 1e-10)


# ----------
#  Training
# ----------
def train():
    prev_time = time.time()
    # --------------
    # Meta-Training
    # --------------
    for epoch in range(opt.n_epochs):
        # 16 Noise
        Tasks = ["Gau0.02", "Gau0.03", "Gau0.04", "Gau0.05", "Gau0.06", "Gau0.07", "Gau0.08", "Gau0.09",
                "GP0.02", "GP0.03", "GP0.04", "GP0.05", "GP0.06", "GP0.07", "GP0.08", "GP0.09"]
        for j, task in enumerate(Tasks):
            train_dataloader = DataLoader(
                ImageDataset_Train(opt.train_data_path, transforms_=transforms_, task_flag=task, ML_mode=opt.ML_mode),
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=opt.n_cpu,)
            for i, batch in enumerate(train_dataloader):
                if i % opt.meta_batch_size == 0:
                    # Save Initial Parameters
                    init_weights_G = deepcopy(generator.state_dict())
                    init_weights_OA = deepcopy(discriminator.state_dict())
                
                data = Variable(batch["image"].type(Tensor))
                label = Variable(batch["label"].type(Tensor))

                # Train Generators
                fake = generator(data)                
                loss_MSE = criterion_MSE(fake, label)
                loss_L1 = criterion_L1(fake, label)
                loss_SSI = criterion_SSI(data, fake)
                loss_G = loss_MSE + 100 * loss_L1 + 10 * loss_SSI

                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()

                # Train Discriminator
                optimizer_OA.zero_grad()
                valid_patch = Variable(Tensor(np.ones((data.size(0), *patch))), requires_grad=False)
                fake_patch = Variable(Tensor(np.zeros((data.size(0), *patch))), requires_grad=False)
                # Real loss
                pred_real = discriminator(label, data)
                loss_real = criterion_MSE(pred_real, valid_patch)
                # Fake loss
                pred_fake = discriminator(fake.detach(), data)
                loss_fake = criterion_MSE(pred_fake, fake_patch)
                # Total loss
                loss_D = 0.5 * (loss_real + loss_fake)

                loss_D.backward()
                optimizer_OA.step()

                # Reptile Update Parameters
                if (i + 1) % opt.meta_batch_size == 0:
                    meta_lr = opt.outer_step_size * (1 - epoch/opt.n_epochs)
                    curr_weights_G = generator.state_dict()
                    curr_weights_OA = discriminator.state_dict()
                    generator.load_state_dict({name: (init_weights_G[name] + meta_lr * 
                                            (curr_weights_G[name] - init_weights_G[name])) for name in curr_weights_G})
                    discriminator.load_state_dict({name: (init_weights_OA[name] + meta_lr * 
                                            (curr_weights_OA[name] - init_weights_OA[name])) for name in curr_weights_OA})
                    
                # Log Progress
                batches_done = epoch * len(train_dataloader) * len(Tasks) + j * len(train_dataloader) + i
                batches_left = opt.n_epochs * len(train_dataloader) * len(Tasks) - batches_done
                time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()

                # Print log
                sys.stdout.write(
                    "\r[Epoch %d/%d] [Task %d/%d] [Batch %d/%d] [Loss G: %f, MSE: %f, L1: %f] [Loss OA: %f] ETA: %s"
                    % (epoch, opt.n_epochs, j, len(Tasks), i, len(train_dataloader), 
                    loss_G.item(), loss_MSE.item(), loss_L1.item(), loss_D.item(), time_left))
                    
    # save model after meta-training
    torch.save(generator.state_dict(), "./saved_models/G_Meta_Train_Done_8Gau8GP.pth")
    
    # load model of meta-training
    #generator.load_state_dict(torch.load(opt.model_save_path))

    # ---------
    # Finetune
    # ---------
    print("\n=== Meta-Training is over, starting finetune! ===")
    for epoch in range(opt.n_FT_epochs):
        train_dataloader = DataLoader(
            ImageDataset_Train(opt.FT_data_path, transforms_=transforms_, ML_mode=0),
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_cpu,)
        for i, batch in enumerate(train_dataloader):
            data = Variable(batch["image"].type(Tensor))
            label = Variable(batch["label"].type(Tensor))

            # Train Generators
            fake = generator(data)                
            loss_MSE = criterion_MSE(fake, label)
            loss_L1 = criterion_L1(fake, label)
            loss_G = loss_MSE + 100 * loss_L1

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            # if i % 3 == 0:
            # Train Discriminator
            optimizer_OA.zero_grad()
            valid_patch = Variable(Tensor(np.ones((data.size(0), *patch))), requires_grad=False)
            fake_patch = Variable(Tensor(np.zeros((data.size(0), *patch))), requires_grad=False)
            # Real loss
            pred_real = discriminator(label, data)
            loss_real = criterion_MSE(pred_real, valid_patch)
            # Fake loss
            pred_fake = discriminator(fake.detach(), data)
            loss_fake = criterion_MSE(pred_fake, fake_patch)
            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)

            loss_D.backward()
            optimizer_OA.step()

            # Log Progress
            batches_done = epoch * len(train_dataloader) + i
            batches_left = opt.n_FT_epochs * len(train_dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Loss G: %f, MSE: %f, L1: %f] [Loss OA: %f] ETA: %s"
                % (epoch, opt.n_FT_epochs, i, len(train_dataloader), 
                loss_G.item(), loss_MSE.item(), loss_L1.item(), loss_D.item(), time_left))

        #==================== Write your own validation code here ====================#
        # if epoch % opt.checkpoint_interval == 0:
        #     YourValidCode()

        # save model
        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            torch.save(generator.state_dict(), "./saved_models/generator_%d.pth" % (epoch))


if __name__ == "__main__":
    train()
