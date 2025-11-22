from model import FrameCorrectionModel

model = FrameCorrectionModel()
model.load_weights("student_weights.h5")  # if available
print("Model ready for fine-tuning.")