# Read pt file

import torch


def read_pt_file(file_path: str):
    data = torch.load(file_path)
    return data


data = read_pt_file("data/10K/database.pt")

print(data.keys())
print(data["key"])
print(data["unique_id"])
print(data["neighbors"])
print(data["dist"])
