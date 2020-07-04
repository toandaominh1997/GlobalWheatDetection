import torch
import torchvision
from engine import train_one_epoch, evaluate
import utils as utils
import pandas as pd
from tqdm import tqdm, trange

from datasets.dataset import WheetDataset

train_df = pd.read_csv('./data/train.csv')
train_dataset = WheetDataset(train_df, phase='train')

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=8, shuffle=True, num_workers=8,
    collate_fn=utils.collate_fn)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=2, pretrained_backbone=True)
model = model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)
num_epochs = 10

for epoch in trange(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=50)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
#     evaluate(model, data_loader_test, device=device)
    state = {
        'epoch': epoch,
        'state_dict': get_state_dict(model)
            }
    torch.save(
            state,
            os.path.join(
                "{}_{}.pth".format('fasterrcnn_resnet50_fpn', epoch)))
