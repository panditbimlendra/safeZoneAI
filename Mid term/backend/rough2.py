model = torch.hub.load('qiuqiangkong/panns_cnn', 'Cnn14', pretrained=True)
model.to(device)
model.eval()

print("Model loaded successfully.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
