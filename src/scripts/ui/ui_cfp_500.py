import sys
import os
import cv2
import torch
import numpy as np
from torchvision import transforms

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QTextEdit,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QProgressBar,
    QMessageBox,
)

from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt

from models.arcface_model import ArcFaceResNet50
from utils.face_detector import create_face_detector
from utils.face_processing import crop_face_with_padding, draw_face_detection_result, select_best_face


# =========================
# 模型加载部分
# =========================

class FaceRecognitionSystem:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize face detector (Improved Haar with better parameters)
        print("Initializing face detector...")
        try:
            self.face_detector = create_face_detector(
                'improved_haar',
                scale_factor=1.1,
                min_neighbors=3,
                min_size=(30, 30)
            )
            print("✓ Face detector initialized successfully")
        except Exception as e:
            print(f"⚠ Warning: Failed to initialize Improved Haar detector: {e}")
            print("  Falling back to standard Haar Cascade...")
            self.face_detector = create_face_detector('haar')
            print("✓ Haar Cascade detector initialized")
        
        # Get project root directory (three levels up from scripts/ui/)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
        
        model_path = os.path.join(project_root, "models", "best_model.pth")
        gallery_path = os.path.join(project_root, "galleries", "gallery_features.pth")

        print("正在加载模型...")
        
        # Load model checkpoint
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Get model config from checkpoint
            num_classes = checkpoint.get('num_classes', 
                                        checkpoint['model_state_dict']['arcface.weight'].shape[0])
            embedding_dim = checkpoint.get('embedding_dim', 512)
            
            # Create model
            self.model = ArcFaceResNet50(
                num_classes=num_classes,
                embedding_dim=embedding_dim,
                pretrained=False
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✓ 模型加载成功: {num_classes} classes, {embedding_dim} dim")
        else:
            print(f"✗ 未找到模型文件: {model_path}")
            raise FileNotFoundError(f"Model not found at {model_path}")

        # Load gallery features
        if os.path.exists(gallery_path):
            self.gallery_features = torch.load(gallery_path, map_location=self.device)
            self.gallery_ids = sorted(self.gallery_features.keys())
            
            # Pre-compute gallery tensor
            gallery_embeddings = []
            for pid in self.gallery_ids:
                emb = self.gallery_features[pid]
                if not isinstance(emb, torch.Tensor):
                    emb = torch.tensor(emb)
                emb = emb.squeeze()
                if emb.dim() == 0:
                    emb = emb.unsqueeze(0)
                gallery_embeddings.append(emb)
            
            self.gallery_tensor = torch.stack(gallery_embeddings).to(self.device)
            print(f"✓ 特征库加载成功: {len(self.gallery_ids)} persons")
        else:
            print(f"✗ 未找到特征库文件: {gallery_path}")
            raise FileNotFoundError(f"Gallery not found at {gallery_path}")
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def infer(self, image_path):
        """
        Face recognition inference with proper face detection
        Returns: similarity score and result image with face box
        """
        try:
            # Read image (support Chinese paths)
            img_array = np.fromfile(image_path, dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if image is None:
                return 0.0, None, f"无法读取图像: {image_path}"

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image_rgb.shape[:2]
            
            print(f"\nImage size: {w}x{h}")
            
            # Step 1: Detect faces using modular detector
            print("Detecting faces...")
            detections = self.face_detector.detect_faces(image_rgb)
            
            if not detections:
                print("⚠ No face detected")
                return 0.0, image, "未检测到人脸"
            
            print(f"✓ Detected {len(detections)} face(s)")
            
            # Step 2: Select the best face (highest confidence)
            best_face = select_best_face(detections, criteria='confidence')
            box = best_face['box']  # [x1, y1, x2, y2]
            confidence = best_face['confidence']
            
            print(f"  Best face: box={box}, confidence={confidence:.2f}")
            
            # Step 3: Crop face with padding
            face_img, original_box = crop_face_with_padding(
                image_rgb, 
                box, 
                padding_ratio=0.2
            )
            x, y, box_w, box_h = original_box
            
            print(f"  Cropped face size: {face_img.shape}")
            
            # Step 4: Preprocess face
            face_tensor = self.transform(face_img).unsqueeze(0).to(self.device)
            
            # Step 5: Extract embedding
            with torch.no_grad():
                embedding = self.model.get_embedding(face_tensor)  # [1, 512]
                embedding = embedding.squeeze(0)  # [512]
            
            # Step 6: Compute similarity with gallery
            similarities = torch.nn.functional.cosine_similarity(
                embedding.unsqueeze(0),  # [1, 512]
                self.gallery_tensor,      # [N, 512]
                dim=1
            )
            
            # Get best match
            max_sim, best_idx = similarities.max(dim=0)
            best_person_id = self.gallery_ids[best_idx.item()]
            similarity_score = max_sim.item()
            
            print(f"✓ Recognition result: ID={best_person_id}, similarity={similarity_score:.4f}")
            
            # Step 7: Draw result on image using utility function
            label_text = f"ID: {best_person_id} ({similarity_score:.2f})"
            result_img = draw_face_detection_result(
                image,
                (x, y, box_w, box_h),
                label=label_text,
                color=(0, 255, 0),
                thickness=2
            )
            
            info = f"识别到人员 ID: {best_person_id}, 相似度: {similarity_score:.4f}"
            
            return similarity_score, result_img, info
            
        except Exception as e:
            print(f"Error during inference: {e}")
            return 0.0, None, f"识别错误: {str(e)}"


# =========================
# UI界面
# =========================

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("人脸伪造识别系统")
        self.setGeometry(200, 100, 1200, 800)

        self.image_path = None

        # 加载系统
        self.system = FaceRecognitionSystem()

        self.init_ui()

    def init_ui(self):

        title = QLabel("人脸伪造识别系统")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("微软雅黑", 24, QFont.Bold))

        # =========================
        # 原图显示
        # =========================
        self.original_label = QLabel()
        self.original_label.setFixedSize(450, 350)
        self.original_label.setStyleSheet(
            "border:2px solid gray;background:white;"
        )
        self.original_label.setAlignment(Qt.AlignCenter)

        original_group = QGroupBox("原始图像")
        original_layout = QVBoxLayout()
        original_layout.addWidget(self.original_label)
        original_group.setLayout(original_layout)

        # =========================
        # 脱敏图显示
        # =========================
        self.result_label = QLabel()
        self.result_label.setFixedSize(450, 350)
        self.result_label.setStyleSheet(
            "border:2px solid gray;background:white;"
        )
        self.result_label.setAlignment(Qt.AlignCenter)

        result_group = QGroupBox("脱敏图像")
        result_layout = QVBoxLayout()
        result_layout.addWidget(self.result_label)
        result_group.setLayout(result_layout)

        left_layout = QVBoxLayout()
        left_layout.addWidget(original_group)
        left_layout.addWidget(result_group)

        # =========================
        # 右侧控制区域
        # =========================

        self.upload_btn = QPushButton("上传照片")
        self.upload_btn.setFixedHeight(50)
        self.upload_btn.setStyleSheet(
            "background:#39b54a;color:white;font-size:18px;"
        )

        self.detect_btn = QPushButton("身份识别")
        self.detect_btn.setFixedHeight(50)
        self.detect_btn.setStyleSheet(
            "background:#2196f3;color:white;font-size:18px;"
        )

        self.progress = QProgressBar()
        self.progress.setValue(0)

        control_group = QGroupBox("操作控制")
        control_layout = QVBoxLayout()
        control_layout.addWidget(self.upload_btn)
        control_layout.addWidget(self.detect_btn)
        control_layout.addWidget(self.progress)
        control_group.setLayout(control_layout)

        # =========================
        # 结果区域
        # =========================

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setFont(QFont("Consolas", 12))

        result_info_group = QGroupBox("识别结果")
        result_info_layout = QVBoxLayout()
        result_info_layout.addWidget(self.result_text)
        result_info_group.setLayout(result_info_layout)

        right_layout = QVBoxLayout()
        right_layout.addWidget(control_group)
        right_layout.addWidget(result_info_group)

        # =========================
        # 总布局
        # =========================

        body_layout = QHBoxLayout()
        body_layout.addLayout(left_layout)
        body_layout.addLayout(right_layout)

        main_layout = QVBoxLayout()
        main_layout.addWidget(title)
        main_layout.addLayout(body_layout)

        self.setLayout(main_layout)

        # =========================
        # 按钮绑定
        # =========================

        self.upload_btn.clicked.connect(self.upload_image)
        self.detect_btn.clicked.connect(self.recognize_identity)

    # =========================
    # 上传照片
    # =========================

    def upload_image(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_path:
            # Verify file can be read
            img_array = np.fromfile(file_path, dtype=np.uint8)
            test_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if test_img is None:
                QMessageBox.warning(self, "错误", f"无法读取图像文件:\n{file_path}")
                return
            
            self.image_path = file_path
            self.show_image(file_path, self.original_label)
            self.result_text.clear()
            self.result_text.append(f"✓ 已上传图片")
            self.result_text.append(f"路径: {file_path}\n")

    # =========================
    # 显示图像
    # =========================

    def show_image(self, image_path, label_widget):

        pixmap = QPixmap(image_path)

        pixmap = pixmap.scaled(
            label_widget.width(),
            label_widget.height(),
            Qt.KeepAspectRatio,
        )

        label_widget.setPixmap(pixmap)

    # =========================
    # 身份识别
    # =========================

    def recognize_identity(self):

        if self.image_path is None:
            QMessageBox.warning(self, "提示", "请先上传图片")
            return

        self.progress.setValue(20)

        similarity, result_img, info = self.system.infer(self.image_path)

        self.progress.setValue(70)

        if result_img is not None:
            # Save to project's temp_results directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
            temp_dir = os.path.join(project_root, "temp_results")
            
            # Create directory if it doesn't exist
            os.makedirs(temp_dir, exist_ok=True)
            
            save_path = os.path.join(temp_dir, "recognition_result.jpg")
            
            # Use cv2.imencode to handle paths properly
            success, encoded_img = cv2.imencode('.jpg', result_img)
            if success:
                with open(save_path, 'wb') as f:
                    f.write(encoded_img.tobytes())

            self.show_image(save_path, self.result_label)

        self.progress.setValue(100)

        # Determine status based on similarity
        if similarity >= 0.8:
            status = "✓ 身份识别一致，系统工作正常"
            color = "green"
        elif similarity >= 0.6:
            status = "⚠ 识别效果良好"
            color = "orange"
        else:
            status = "✗ 识别效果较差"
            color = "red"

        self.result_text.clear()

        self.result_text.append("=" * 40)
        self.result_text.append("人脸识别结果")
        self.result_text.append("=" * 40)
        self.result_text.append(f"\n{info}\n")
        self.result_text.append(f"判别结果: {status}\n")
        
        self.result_text.append("评估标准:")
        self.result_text.append("- 优秀: 相似度 > 0.8")
        self.result_text.append("- 良好: 相似度 0.6 ~ 0.8")
        self.result_text.append("- 需改进: 相似度 < 0.6")


# =========================
# 主函数
# =========================

if __name__ == '__main__':

    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
