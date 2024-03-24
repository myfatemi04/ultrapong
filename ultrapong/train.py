import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import cv2

class fastdet(torch.nn.Module):
    def __init__(self):
        super(fastdet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5, padding=2)
        self.conv2 = torch.nn.Conv2d(6, 1, 5, padding=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

cap = cv2.VideoCapture("video.mp4")
import json
labels = json.load(open("labels.json"))

label_counter = 0
counter = 0

frames = []
labels_existent = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    counter += 1
    if (counter % 10) != 0:
        continue

    if label_counter >= len(labels):
        continue

    label = labels[label_counter]
    if label is not None:
        frames.append(frame)
        labels_existent.append(label)

    label_counter += 1

cap.release()

print("generated quick dataset")

frames_tensor = torch.tensor(np.array(frames))

print("frames tensor", frames_tensor.shape)
print("labels", len(labels_existent))

dataset = torch.utils.data.TensorDataset(frames_tensor, torch.tensor(labels_existent))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

model = fastdet()
optim = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for i, (frame, label) in enumerate(dataloader):
        print("Epoch", epoch, "Batch", i, "Frame", frame.shape, "Label", label)
        frame = frame.permute(0, 3, 1, 2).float() / 255
        pred = model(frame)
        # cross entropy loss
        target_mask = torch.zeros((len(pred), 1, frame.shape[2], frame.shape[3]))
        for j in range(len(label)):
            target_mask[j, 0, label[j, 1], label[j, 0]] = 1
        target_mask = target_mask.view(-1, frame.shape[2] * frame.shape[3]).float()
        pred0 = pred[0]
        pred = pred.view(-1, frame.shape[2] * frame.shape[3])

        loss = F.binary_cross_entropy_with_logits(pred, target_mask, reduction="none")
        loss = loss[target_mask == 1] + loss[target_mask == 0].mean(dim=-1, keepdim=False)
        print("Loss", loss.item())

        cv2.imshow("pred", pred0.permute(1, 2, 0).detach().numpy())
        cv2.waitKey(1)

        optim.zero_grad()
        loss.backward()
        optim.step()

