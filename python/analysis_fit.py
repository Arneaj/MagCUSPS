import pandas as pd
import numpy as np

df_inputs = pd.read_csv( "../.result_folder/great_analysis867056.csv", sep="," )
df_inputs2 = pd.read_csv( "../.result_folder/result_print_csv.csv", sep="," )

shue_fit_loss = df_inputs["Shue97_fit_loss"]
liu_fit_loss = df_inputs["Liu12_fit_loss"]
rolland_fit_loss = df_inputs["Rolland25_fit_loss"]

shue_grad_J_fit_over_ip = df_inputs["Shue97_grad_J_fit_over_ip"]
liu_grad_J_fit_over_ip = df_inputs["Liu12_grad_J_fit_over_ip"]
rolland_grad_J_fit_over_ip = df_inputs["Rolland25_grad_J_fit_over_ip"]

shue_grad_J_fit_over_ip = df_inputs["Shue97_grad_J_fit_over_ip"]
liu_grad_J_fit_over_ip = df_inputs["Liu12_grad_J_fit_over_ip"]
rolland_grad_J_fit_over_ip = df_inputs["Rolland25_grad_J_fit_over_ip"]

ip_avg_std_dev = df_inputs2["ip_avg_std_dev"]

avg_shue_fit_loss = shue_fit_loss.mean()
avg_liu_fit_loss = liu_fit_loss.mean()
avg_rolland_fit_loss = rolland_fit_loss.mean()

avg_shue_grad_J_fit_over_ip = shue_grad_J_fit_over_ip.mean()
avg_liu_grad_J_fit_over_ip = liu_grad_J_fit_over_ip.mean()
avg_rolland_grad_J_fit_over_ip = rolland_grad_J_fit_over_ip.mean()

avg_ip_avg_std_dev = ip_avg_std_dev.mean()

print("avg_ip_avg_std_dev:", avg_ip_avg_std_dev)

print("avg_shue_fit_loss:", avg_shue_fit_loss)
print("avg_shue_grad_J_fit_over_ip:", avg_shue_grad_J_fit_over_ip)

print("avg_liu_fit_loss:", avg_liu_fit_loss)
print("avg_liu_grad_J_fit_over_ip:", avg_liu_grad_J_fit_over_ip)

print("avg_rolland_fit_loss:", avg_rolland_fit_loss)
print("avg_rolland_grad_J_fit_over_ip:", avg_rolland_grad_J_fit_over_ip)
