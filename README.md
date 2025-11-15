# Dự án Phân loại Chữ số Viết tay (Digits Classification)

Dự án này là một ví dụ cơ bản về cách xây dựng, huấn luyện và đánh giá một mô hình A.I để nhận dạng chữ số viết tay. Chúng tôi sử dụng bộ dữ liệu MNIST và thư viện PyTorch để thiết kế, huấn luyện mô hình và đánh giá mô hình.

##  Mục tiêu dự án

* Xây dựng một mô hình AI có khả năng nhận dạng chữ số viết tay (từ 0 đến 9) với độ chính xác cao.

* Tìm hiểu quy trình làm việc cơ bản của một dự án Deep Learning với PyTorch, bao gồm các giai đoạn: tải dữ liệu, tiền xử lý, thiết kế mô hình, huấn luyện (training) và đánh giá (evaluation).
## Kế hoạch thực hiện dự án
<img width="1920" height="1080" alt="grant" src="https://github.com/user-attachments/assets/d2820dc9-db0b-49b4-9d7d-bbdcfd951a7b" />

---

##  Yêu cầu cài đặt (Requirements)

Trước khi cài đặt, bạn phải cài đặt trước **Python 3.10**, một virtual environment **(Khuyến khích sử dụng [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) )**

 

## Hướng dẫn cài đặt (BẮT BUỘC)

### Bước 1: Thiết lập môi trường

Trong terminal, tạo một virtual environment mới và kích hoạt. Đối với môi trường do micromamba tạo ra thì: 

```bash

micromamba create -n NAME python=3.10
micromamba activate NAME

```

### Bước 2: Cài requirement.txt

**Trong cùng một cái virtual environment**, **cd vào thư mục dự án và chạy**:

```bash

pip install -r requirement.txt

```

Để tải dự án này, bạn cần có **Python 3.10** và virtual environment **(pyenv, micromamba, mamba,...)**. 

## Cách chạy huấn luyện và đánh giá mô hình.

## Changelog 15/11/2025
-Changelogs in  dataloader.py 
	* Added dataset methods (__len__, __getitem__).
-Changelogs in trainer.py: 
	* Split full dataset into three subsets: train, validation, test with ratio respectively 6:2:2.
	* Immplemented dataloader for each subset (No data collator, for now).
	* Added dataloader test code.
	
