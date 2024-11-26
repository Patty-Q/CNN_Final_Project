import os  
from PIL import Image  
import torch  
from models.model import get_net  
import torchvision.transforms as transforms  
from config import config  
  
  
  
# Define a function that handles a single image 
def identify_single_image(image_path, model, device, transform):  
    image = Image.open(image_path).convert('RGB')  
    image = transform(image).unsqueeze(0).to(device)  
    with torch.no_grad():  
        outputs = model(image)  
        probs = torch.nn.functional.softmax(outputs[0], dim=0)  
        _, predicted = torch.max(probs, 0)  
    return predicted.item()  
  
# Define functions that handle directories
def process_images_in_directory(directory_path, model, device, transform, class_labels):  
    for filename in os.listdir(directory_path):  
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  
            file_path = os.path.join(directory_path, filename)  
            predicted_class = identify_single_image(file_path, model, device, transform)  
            print(f'File: {filename}, Predicted class: {predicted_class}, Label: {class_labels[predicted_class]}')  
  
# Usage example
image_directory = './detect'  # Picture directory path
weight_path = "./checkpoints/best_model/resnet50/1/model_best.pth.tar"  # Weight file path

# tag list  
class_labels = class_labels = ["苹果健康","苹果黑星病一般","苹果黑星病严重","苹果灰斑病","苹果雪松锈病一般",
               "苹果雪松锈病严重","樱桃健康","樱桃白粉病一般","樱桃白粉病严重","玉米健康",
               "玉米灰斑病一般","玉米灰斑病严重","玉米锈病一般","玉米锈病严重","玉米叶斑病一般",
               "玉米叶斑病严重","玉米花叶病毒病","葡萄健康","葡萄黑腐病一般","葡萄黑腐病严重",
               "葡萄轮斑病一般","葡萄轮斑病严重","葡萄褐斑病一般","葡萄褐斑病严重","柑桔健康",
               "柑桔黄龙病一般","柑桔黄龙病严重","桃健康","桃疮痂病一般","桃疮痂病严重","辣椒健康",
               "辣椒疮痂病一般","辣椒疮痂病严重","马铃薯健康","马铃薯早疫病一般","马铃薯早疫病严重",
               "马铃薯晚疫病一般","马铃薯晚疫病严重","草莓健康","草莓叶枯病一般","草莓叶枯病严重",
               "番茄健康","番茄白粉病一般","番茄白粉病严重","番茄早疫病一般",
               "番茄早疫病严重","番茄晚疫病菌一般","番茄晚疫病菌严重","番茄叶霉病一般","番茄叶霉病严重",
               "番茄斑点病一般","番茄斑点病严重","番茄斑枯病一般","番茄斑枯病严重","番茄红蜘蛛损伤一般",
               "番茄红蜘蛛损伤严重","番茄黄化曲叶病毒病一般","番茄黄化曲叶病毒病严重","番茄花叶病毒病"]  
  
# Load model  
model = get_net()  
model = torch.nn.DataParallel(model)  
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
model.load_state_dict(torch.load(weight_path)['state_dict'])  
model.to(device)  
model.eval()  
  
# Image conversion 
transform = transforms.Compose([  
    transforms.Resize((256, 256)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])
process_images_in_directory(image_directory, model, device, transform, class_labels)