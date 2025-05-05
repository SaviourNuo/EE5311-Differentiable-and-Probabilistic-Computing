import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import io

# Parse and read the data in 'data1.txt'
def parse_data_file(filename):
    with open(filename, "r") as f:
        data_str = f.read()
    
    records_raw = data_str.split("---")
    records = []
    
    for rec in records_raw:
        rec = rec.strip()
        if not rec:
            continue
        
        lines = rec.split("\n")
        doseA, doseB = None, None
        csv_start_index = 0
        
        for i, line in enumerate(lines):
            if "drug A:" in line:
                doseA = float(line.split(":")[1].strip())
            elif "drug B:" in line:
                doseB = float(line.split(":")[1].strip())
            elif line.startswith("time"):
                csv_start_index = i
                break
        
        if csv_start_index == 0:
            continue
        
        df = pd.read_csv(io.StringIO("\n".join(lines[csv_start_index:])))
        times = df["time"].values
        platelets = df["platelet"].values
        
        records.append((doseA, doseB, times, platelets))
    
    return records

# Parse the data file
records = parse_data_file("data1.txt")

# Define the ODE function using a neural network
class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 8),
            nn.Tanh(),
            nn.Linear(8, 8),
            nn.Tanh(),
            nn.Linear(8, 8),
            nn.Tanh(),
            nn.Linear(8, 1)
        )

    def forward(self, t, state):
        p, A, B = state[:, 0], state[:, 1], state[:, 2]
        inputs = torch.stack((t.repeat(p.shape[0]), A, B, p), dim=1)
        dp_dt = self.net(inputs).squeeze(-1)
        
        return torch.cat((dp_dt.unsqueeze(-1), torch.zeros_like(A).unsqueeze(-1), torch.zeros_like(B).unsqueeze(-1)), dim=1)

# Initialize the model and move it to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ODEFunc().to(device)

# Define the loss function
def compute_loss(model, records, lambda_phy=0.1):
    loss_data = 0.0
    loss_physics = 0.0

    for doseA, doseB, times, platelets in records:
        u0 = torch.tensor([[platelets[0], doseA, doseB]], dtype=torch.float32, device=device)
        times_torch = torch.tensor(times, dtype=torch.float32, device=device, requires_grad=True)

        sol = odeint(model, u0, times_torch, method='dopri5')
        p_pred = sol[:, 0, 0]

        # Data loss (MSE)
        loss_data += torch.sum((p_pred - torch.tensor(platelets, dtype=torch.float32, device=device)) ** 2)

        # Physics loss (MSE)
        dp_dt_pred = torch.autograd.grad(p_pred, times_torch, grad_outputs=torch.ones_like(p_pred), create_graph=True)[0]
        inputs = torch.stack((times_torch, doseA * torch.ones_like(times_torch), doseB * torch.ones_like(times_torch), p_pred), dim=1)
        dp_dt_model = model.net(inputs).squeeze(-1)  

        loss_physics += torch.sum((dp_dt_pred - dp_dt_model) ** 2)

    total_loss = loss_data + lambda_phy * loss_physics
    return total_loss

print("Loss before training: ", compute_loss(model, records).item())

# Train the model using the Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1000):
    optimizer.zero_grad()
    loss = compute_loss(model, records)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}: Loss = {loss.item()}")

print("Loss after training: ", loss.item())

# Predict the platelet levels for a new set of drug dosages
test_doseA, test_doseB = 2.1, 2.4
u0_test = torch.tensor([[1.0, test_doseA, test_doseB]], dtype=torch.float32, device=device)
test_times = torch.arange(0, 10.5, 0.5, dtype=torch.float32, device=device)

sol_test = odeint(model, u0_test, test_times, method='dopri5')
p_pred_test = sol_test[:, 0, 0].cpu().detach().numpy()

# Save the results to a CSV file
with open("results(PINN).csv", "w") as f:
    f.write("Daily dosage:\n")
    f.write(f"  drug A: {test_doseA}\n")
    f.write(f"  drug B: {test_doseB}\n\n")
    f.write("time,platelet\n")
    for t, p in zip(test_times.cpu().numpy(), p_pred_test):
        f.write(f"{t},{p}\n")

print("Results saved in 'results(PINN).csv'")

# Plot the results
plt.figure(figsize=(8, 5))

for doseA, doseB, times, platelets in records:
    plt.plot(times, platelets, label=f"A={doseA}, B={doseB}", lw=1.5)

plt.plot(test_times.cpu().numpy(), p_pred_test, label="Predicted (A=2.1, B=2.4)",
         color="black", linestyle="dashed", linewidth=3)

plt.xlabel("Time (months)")
plt.ylabel("Relative Platelet Level")
plt.title("Platelet Level Over Time for Different Drug Dosages")
plt.legend(fontsize=8, loc="upper left")
plt.grid(True)
plt.savefig("platelet_prediction_PINN.png", dpi=300)
plt.show()

