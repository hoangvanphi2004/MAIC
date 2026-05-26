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

Ngoài ra, để chạy mô hình đã được lưu ta có thể sử dụng câu lệnh:

```bash
python refactor/inference.py
```

### Điều chỉnh training config

Tùy vào loại môi trường, mô hình và các thông số mà ta cần huấn luyện mà ta có thể thay đổi các thông số tại file `refactor\train_config.json`

| Tên tham số      | Mô tả |
  --- | --- |
  | env_id | Thể hiện tên loại môi trường mà ta muốn huấn luyện, tên các loại môi trường có thể được tìm kiếm tại file `refactor\multigrid\multigrid\envs\__init__.py` |
  | num_agents | Số lượng tác tử được sử dụng trong môi trường |
  | obs_state_mode | Tùy chỉnh số chiều trạng thái, với `full` thể hiện số chiều trạng thái cao và `simple` thể hiện số chiều trạng thái thấp |
  | seed | Seed huấn luyện mô hình |
  | episodes | Số lượng lần huấn luyện (episodes) trong cả quá trình huấn luyện |
  | steps_per_episode | Số bước thời gian tối đa mà các tác tử có thể tương tác với môi trường |
  | replay_size | Độ lớn của bộ đệm phát lại (Replay Buffer) |
  | batch_size | Độ lớn của một lô dữ liệu (batch) trong quá trình huấn luyện |
  | start_steps | Số bước thời gian (timestep) chọn hành động ngẫu nhiên kể từ khi bắt đầu huấn luyện |
  | steps_per_update | Tần suất huấn luyện |
  | updates_num | Số lô dữ liệu được huấn luyện trong một lần huấn luyện |
  | save_every | Tấn suất lưu dữ liệu tính theo episode |
  | model_dir | Đường dẫn thư mục lưu mô hình |
  | render | Bật/Tắt hiện video thể hiện quá trình các tác tử tương tác trong quá trình huấn luyện |
  | record_video | Bật/Tắt Lưu video thể hiện quá trình tương tác trong quá trình huấn luyện |
  | video_every | Tấn suất lưu vieo thể hiện quá trình tương tác tính theo episode |
  | plot_every | Tần suất vẽ biểu đồ của cấc thông số trong quá trình huấn luyện |
  | scaled_information_gain_coef | Hằng số thể hiện độ ảnh hưởng của lượng thông tin thu thập đến quá trình huấn luyện của mô hình, hằng số này bằng 0 có nghĩa là hoàn toàn không dùng lượng thông tin thu thập |
  | scaled_entropy_coef | Hằng số thể hiện độ ảnh hưởng của entropy đến quá trình huấn luyện của mô hình, hằng số này bằng 0 có nghĩa là hoàn toàn không sử dụng entropy |
  | model_config | Các thông số của mô hình |

Các thông số của mô hình được mô tả ở bảng sau:

| Tên tham số      | Mô tả |
  --- | --- |
  | hidden_dim | Kích thước các chiều ẩn của Actor và Critic |
  | lr | Tốc độ huấn luyện (learning rate) |
  | gamma | Giá trị tham số $\gamma$ |
  | tau | Giá trị tham số $\tau$ |
  | alpha1 | Giá trị ban đầu của tham số $\alpha_1$ |
  | alpha2 | Giá trị ban đầu của tham số $\alpha_2$ |
  | alpha_kl | Giá trị của tham số $\alpha_{KL}$ |
  | policy_update_steps | Số lần cập nhật mỗi lần cập nhật Actor |
  | auto_entropy_tuning | Bật/Tắt tự động điều chỉnh $\alpha_1$ và $\alpha_2$ |
  | target_entropy_scale | Giá trị $\beta$ trong $\mathcal{\bar{H}} = \beta \log(n\|A\|)$ |

### Điều chỉnh inference config

Để tùy chỉnh quá trình chạy mô hình đã được lưu, ta có thể điều chỉnh các trường trong file `inference_config.json`

| Tên tham số      | Mô tả |
  --- | --- |
  | env_id | Thể hiện tên loại môi trường ta muốn chạy |
  | num_agents | Số lượng tác tử trong môi trường |
  | obs_state_mode | Tùy chỉnh số chiều trạng thái, với `full` thể hiện số chiều trạng thái cao và `simple` thể hiện số chiều trạng thái thấp |
  | checkpoint | Checkpoint của mô hình đã lưu |
  | episodes | Số lần chạy của mô hình |
  | save_video | Đường dẫn thư mục để lưu video thể hiện quá tương tác của tác tử |
  | fps | Số khung hình một giây của video thể hiện quá tương tác của tác tử |
  | steps_per_episode | Số bước thời gian (timestep) mỗi episode |

## Dữ liệu đầu ra của quá trình huấn luyện

Các dữ liệu về hiệu năng của mô hình sẽ được thu thập vào file `metrics_latest.json`, file này sẽ gồm các trường dữ liệu được thể hiện ở bảng bên dưới
  
| Trường dữ liệu      | Mô tả |
  --- | --- |
  | episode_rewards | Tổng phần thưởng thu thập được mỗi episode |
  | episode_lengths | Số bước thời gian (timestep) của mỗi episode |
  | actor_losses | Giá trị trung bình mất mát của Actor theo mỗi episode |
  | critic_losses | Giá trị trung bình mất mát của Critic theo mỗi episode |
  | alpha1_losses | Giá trị trung bình mất mát của $\alpha_1$ theo mỗi episode |
  | alpha2_losses | Giá trị trung bình mất mát của $\alpha_2$ theo mỗi episode |
  | q_values | Giá trị trung bình của hàm giá trị theo mỗi episode |
  | entropies | Giá trị trung bình của độ bất định theo mỗi episode |
  | alpha1_values | Giá trị trung bình của $\alpha_1$ theo mỗi episode |
  | alpha2_values | Giá trị trung bình của $\alpha_2$ theo mỗi episode |
  | information_gains | Giá trị trung bình của lượng thông tin thu thập theo mỗi episode |
  | kl_divergences | Giá trị trung bình của KL divergence giữa chính sách mới và chính sách cũ theo mỗi episode |
  | unique_states_seen | Số lượng trạng thái mà tác tử đã đi qua tính đến 1 episode nhất định |

Ngoài ra dữ liệu đầu ra còn có mô hình được lưu lại (pth file) và video thể hiện quá trình tương tác được lưu lại với tần suất được định nghĩa trong config.

## Kết quả

<p align="center"><img width="512" height="512" alt="inference" src="https://github.com/user-attachments/assets/5bf30e74-1f8b-4c40-b72f-9bc923502981" /></p>

<img width="1130" height="667" alt="image" src="https://github.com/user-attachments/assets/ac91e98b-7cf3-480c-98fa-7e5d45a48a98" />

<img width="1131" height="665" alt="image" src="https://github.com/user-attachments/assets/e9ab459e-209f-4fbc-b373-9800fe16a79c" />


