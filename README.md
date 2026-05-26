<div align="center"> 

  # TĂNG CƯỜNG KHẢ NĂNG KHÁM PHÁ TRONG HỌC TĂNG CƯỜNG ĐA TÁC TỬ DỰA TRÊN CỰC ĐẠI HOÁ LƯỢNG THÔNG TIN THU THẬP

</div>

## Tổng Quan
Học tăng cường đa tác tử hiện này là một lĩnh vực nghiên cứu quan trọng, với nhiều ứng dụng thực tiễn như điều khiển robot phối hợp, quản lý giao thông thông minh hay quản lý giám sát khu vực bằng thiết bị bay không người lái. Để huấn luyện được một mô hình học tăng cường đa tác tử hiệu quả, mô hình cần có khả năng khám phá môi trường một cách một cách hiệu quả. Từ đó, bài khóa luận của em đề xuất một mô hình tăng cường khả năng khám phá cho môi trường đa tác tử sử dụng lượng thông tin thu thập. Repo này sẽ chứa code để huấn luyện mô hình đề xuất (MAIC), COMA và MASAC và lưu lại dữ liệu cần thiết cho quá trình vẽ các biểu đồ. 
## Cách Cài Đặt
### Cách Cài Đặt Tổng Quan
Đầu tiên, để dự án có thể chạy được, ta cần cài đặt phiên bản mới nhất của python, phiên bản mới nhất của python có thể được tải xuống [tại đây](https://www.python.org/downloads/) 

Sau đó, ta sẽ cần tải xuống pytorch. Lưu ý, để huấn luyện được các tác tử trên gpu ta cần phải tải phiên bản pytoch sử dụng cuda. Pytorch có thể được cài đặt theo hướng dẫn [tại đây](https://pytorch.org/get-started/locally/)

Tiếp theo, ta sẽ cài đặt gymnasium sử dụng câu lệnh pip bên dưới, phiên bản được sử dụng trong các thí nghiệm là 1.1.1.

```bash
pip install gymnasium==1.1.1
```

Tiếp theo đó, ta sẽ cài đặt môi trường multigrid đã được tinh chỉnh:

```bash
pip install -e /refactor/multigrid
```

Cuối cùng, ta có thể huấn luyện mô hình sử dụng câu lệnh:

```bash
python /refactor/main.py 
```

## Lời Cảm Ơn
## Liên Hệ
