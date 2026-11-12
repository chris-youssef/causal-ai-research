# 1. The Ice Cream and Shark Attacks (The Confounder Test)
# The Setup: Data shows a strong statistical correlation between ice cream sales at a beach snack bar and 
# the number of shark attacks in the water.
# The Causal Question: Does eating ice cream cause shark attacks?
# The Causal Graph:
# Variable X (Ice Cream Sales) → Variable Y (Shark Attacks)
# Variable Z (Temperature / Sunny Weather) → X and Z → Y [1]
# How to Test It: Feed the data and the known confounder (Z = Temperature) into your Causal AI tool. 
# A correct causal model will determine that the direct effect of ice cream on shark attacks is zero once you adjust for temperature 
# (the common cause). [1]

import pyro
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from torch.distributions import constraints
import pyro.poutine as poutine


# Generate synthetic data

np.random.seed(1)

n = 500

temperature = np.random.normal(80, 10, n)
ice_cream_sales = (temperature / 2) + np.random.normal(0,2,n)

shark_rate = (temperature -50) / 30
shark_rate = np.clip(shark_rate, 0.05, None)
shark_attacks = np.random.poisson(shark_rate)


# Put the generated data into a DataFrame

df = pd.DataFrame({
    "temperature": temperature,
    "ice_cream_sales": ice_cream_sales,
    "shark_attacks": shark_attacks
})

print(df.head(10).round(0))


# Check the correlations between the three variables

ice_cream_shark_attack_correlation = df["ice_cream_sales"].corr(df["shark_attacks"])
temperature_shark_attack_correlation = df["temperature"].corr(df["shark_attacks"])
ice_cream_temperature_correlation = df["ice_cream_sales"].corr(df["temperature"])
print("The correlation between ice cream sales and shark attacks is:", ice_cream_shark_attack_correlation)
print("The correlation between temperature and shark attacks is:", temperature_shark_attack_correlation)
print("The correlation between ice cream sales and temperature is:", ice_cream_temperature_correlation)

# plt.scatter(df["ice_cream_sales"], df["shark_attacks"])
# plt.xlabel("Ice Cream Sales")
# plt.ylabel("Shark Attacks")
# plt.show()


# Remove the known effect of temperature and check the remaining correlation

ice_cream_residuals = ice_cream_sales - (temperature / 2)
print(ice_cream_residuals[:10])

shark_residuals = shark_attacks - shark_rate

residual_correlation = np.corrcoef(ice_cream_residuals, shark_residuals)[0, 1]
print("Residual correlation:", residual_correlation)


# Convert the data to PyTorch tensors for Pyro

temperature_tensor = torch.tensor(temperature, dtype=torch.float32)
ice_cream_tensor = torch.tensor(ice_cream_sales, dtype = torch.float32)
shark_tensor = torch.tensor(shark_attacks, dtype = torch.float32)


# Build the Pyro model and learn how temperature affects both variables

def model():
    temperature_effect = pyro.param(
        "temperature_effect",
        torch.tensor(0.2)
    )
    predicted_ice_cream = temperature_tensor * temperature_effect

    shark_temperature_effect = pyro.param(
        "shark_temperature_effect",
        torch.tensor(0.01),
        constraint=constraints.positive)

    predicted_shark_rate = (temperature_tensor - 50) * shark_temperature_effect
    predicted_shark_rate = torch.clamp(predicted_shark_rate, min = 0.05)

    with pyro.plate("data", n):
        pyro.sample(
            "ice_cream",
            dist.Normal(predicted_ice_cream, 2),
            obs=ice_cream_tensor)
        pyro.sample(
            "shark_attacks",
            dist.Poisson(predicted_shark_rate),
            obs=shark_tensor
)


# Set up and train the model with SVI

optimizer = pyro.optim.Adam({"lr": 0.01})

def guide():
    pass

loss = Trace_ELBO()

svi = SVI(
    model=model,
    guide=guide,
    optim=optimizer,
    loss=loss
)
print()
for i in range(1000):
    current_loss = svi.step()
    if i % 100 == 0:
        print("Step:", i, "Current loss:", current_loss)

print("Learned Temperature -> Ice Cream Effect:", pyro.param("temperature_effect").item())
print("Learned Temperature -> Shark Attack Effect:", pyro.param("shark_temperature_effect").item())

print()


# Build a causal model that can be used with Pyro's do-operator

def causal_model():
    temperature = pyro.sample(
        "temperature",
        dist.Normal(80.0, 10.0)
    )

    temperature_effect = pyro.param("temperature_effect")
    predicted_ice_cream = temperature * temperature_effect

    ice_cream = pyro.sample(
        "ice_cream",
        dist.Normal(predicted_ice_cream, 2)
    )

    shark_effect = pyro.param("shark_temperature_effect")
    predicted_shark_rate = (temperature - 50) * shark_effect
    predicted_shark_rate = torch.clamp(predicted_shark_rate, min=0.05)

    shark_attacks = pyro.sample(
        "shark_attacks",
        dist.Poisson(predicted_shark_rate)
    )

    return ice_cream, shark_attacks, predicted_shark_rate


# Temperature interventions: changing temperature should affect both outcomes

temperature_70_model = poutine.do(
    causal_model,
    data={"temperature": torch.tensor(70.0)}
)

ice_cream_70, sharks_70, shark_rate_70 = temperature_70_model()

print()
print("Pyro do(Temperature = 70)")
print("Ice cream sales:", ice_cream_70.item())
print("Expected shark attack rate:", shark_rate_70.item())
print("Sampled shark attacks:", sharks_70.item())


temperature_100_model = poutine.do(
    causal_model,
    data={"temperature": torch.tensor(100.0)}
)

ice_cream_100, sharks_100, shark_rate_100 = temperature_100_model()

print()
print("Pyro do(Temperature = 100)")
print("Ice cream sales:", ice_cream_100.item())
print("Expected shark attack rate:", shark_rate_100.item())
print("Sampled shark attacks:", sharks_100.item())


# Ice cream interventions: hold temperature constant and change only ice cream sales

ice_cream_60_model = poutine.do(
    causal_model,
    data={
        "temperature": torch.tensor(80.0),
        "ice_cream": torch.tensor(60.0)
    }
)

ice_cream_60, sharks_60, shark_rate_60 = ice_cream_60_model()

print()
print("Pyro do(Temperature = 80, Ice Cream = 60)")
print("Ice cream sales:", ice_cream_60.item())
print("Expected shark attack rate:", shark_rate_60.item())
print("Sampled shark attacks:", sharks_60.item())


ice_cream_80_model = poutine.do(
    causal_model,
    data={
        "temperature": torch.tensor(80.0),
        "ice_cream": torch.tensor(80.0)
    }
)

ice_cream_80, sharks_80, shark_rate_80 = ice_cream_80_model()

print()
print("Pyro do(Temperature = 80, Ice Cream = 80)")
print("Ice cream sales:", ice_cream_80.item())
print("Expected shark attack rate:", shark_rate_80.item())
print("Sampled shark attacks:", sharks_80.item())
