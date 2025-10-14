from backend.reco.trainer import train

if __name__ == "__main__":
    stats = train()
    print("Training complete:", stats)
