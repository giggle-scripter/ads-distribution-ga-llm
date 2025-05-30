# Advertising Distribution Optimization with LLM-Enhanced Genetic Algorithm
Dự án này triển khai thuật toán di truyền tiên tiến được tăng cường bằng Mô hình ngôn ngữ lớn (LLM) để giải quyết các vấn đề tối ưu hóa phân phối quảng cáo. 

Hệ thống này phân phối quảng cáo cho các vị trí trên bảng quảng cáo trong khi tối đa hóa doanh thu, số lượng quảng cáo được gán và tuân thủ các ràng buộc kinh doanh khác nhau (như conflict).

## Bài toán
Các đối tượng xuất hiện trong bài toán
- **Billboards**: Các biển quảng cáo, có từ 1-6 mặt (slots)
- **Advertisements**: Các quảng cáo, chứa thông tin về base price (giá sẽ trả cho billboard company ứng với billboard có 6 mặt) và max budget (số tiền lớn nhất mà người muốn gắn quảng cáo sẽ trả cho billboard company).
- **Conflict**: Các cặp quảng cáo không thể cùng đặt cùng billboard do là đối thủ cạnh tranh.
- **Constraints**: 
  - Các ràng buộc cứng: Conflict (không có 2 cặp conflict nào được gán cùng 1 billboard), Unique (mỗi quảng cáo xuất hiện đúng 1 lần trên toàn bộ billboard).
  - Các ràng buộc mềm: Max budget (một quảng cáo nếu có chi phí thực tế lớn hơn max budget thì sẽ chỉ trả max budget cho công ty quảng cáo đó).
- **Objectives**: Tối đa hóa doanh thu của công ty cung cấp biển quảng cáo (revenue)
### Revenue Model
Doanh thu được tính bằng cách sử dụng các hệ số nhân dựa trên số slots của biển quảng cáo:
- 1 slot: 2.0× multiplier (premium placement)
- 2 slots: 1.7× multiplier
- 3 slots: 1.5× multiplier
- 4 slots: 1.3× multiplier
- 5 slots: 1.1× multiplier
- 6 slots: 1.0× multiplier (standard rate)

Để chỉnh sửa các thông số này, bạn có thể thay đổi các giá trị trong `problem.py`, phần `SLOTS_FACTOR` trong class `Problem`.

## Thuật toán
Trong dự án này, nhóm đưa ra 2 thuật toán chính để giải bài toán:
### Genetic Algorithm (GA)
Thuật toán GA được sử dụng với
- Cá thể: Được mã hóa dưới dạng một mảng sol, trong đó `sol[i]=j` nghĩa là slot i được gán quảng cáo j. Nếu `sol[i]=-1` thì slot i không được gán quảng cáo nào
- Lai ghép: Sử dụng phép lai ghép Uniform Crossover, nghĩa là duyệt mỗi vị trí và xem xác suất để lựa chọn giá trị tại vị trí đó theo cha hay mẹ.
- Đột biến: Chọn 1 vị trí (slot) và unassign (đưa về chưa gán) hoặc assign new.
- Selection: Sử dụng Top-K Selection để chọn cá thể cha mẹ.
- Replacement: Luôn giữ lại một vài cá thể tốt nhất của quần thể cũ.
### LLM-GA
Sử dụng LLM như là một phương pháp để thoát khỏi tối ưu cục bộ. Quy trình tại mỗi vòng generation vẫn giống hệt GA. Tuy nhiên nếu sau một vài thế hệ liên tiếp không cải thiện được lời giải, sẽ:
1. Chọn 1 vài cá thể theo chiến lược nào đó từ quần thể (sau bước replacement).
2. Với mỗi cá thể, xây dựng prompt tương ứng với problem context và solution.
3. Gửi prompt và yêu cầu LLM trả về một/một chuỗi các transformation để biến đổi lời giải hiện có thành một lời giải mới.
4. Thêm các cá thể mới vào quần thể rồi thực hiện sắp xếp và lựa chọn.

Các transformation được cung cấp:
- `unassigned`: Unassign một vài vị trí.
- `assign-new`: Assign một slot với 1 ad mới.
- `swap-assign`: Chọn 2 vị trí và hoán đổi giá trị gán tại 2 vị trí đó (hoán đổi 2 ad).
- `swap-billboard`: Chọn 2 billboard và hoán đổi tất cả các ad trên 2 billboard (nếu số mặt khác nhau thì một vài ad trên billboard nhiều mặt hơn được giữ lại.)

## Kiến trúc hệ thống
### Core Modules

#### 1. `problem.py` - Problem Definition
- **Problem class**: Một lớp bao gồm tất cả dữ liệu và ràng buộc của bài toán
- **Evaluation methods**: Các phương thức tính toán vi phạm, doanh thu và số lượng quảng cáo được gán
- **Data loading**: Hỗ trợ nhập dữ liệu từ file hoặc console
- **Random generation**: Hỗ trợ tạo các trường hợp bài toán thử nghiệm 

#### 2. `evo.py` - Genetic Algorithm Engine
- **Individual class**: Biểu diễn các cá thể với cấu trúc gen và đánh giá fitness
- **Population class**: Quản lý tập hợp các cá thể
- **Genetic operators**: Crossover, mutation, selection mechanisms
- **GA implementations**: Các thuật toán di truyền tiêu chuẩn và thuật toán di truyền tăng cường LLM

#### 3. `llm_support.py` - LLM Integration
- **PromptBuilder**: Class hỗ trợ tạo các prompt có cấu trúc cho LLM
- **LLMSupporter**: Quản lý các tương tác với LLM và phân tích phản hồi
- **Transformations**: Thực hiện áp dụng các cải tiến giải pháp được gợi ý bởi LLM

#### 4. `test.py` - Application Entry Point
- **Configuration**: Thiết lập các tham số thuật toán và cấu hình LLM
- **Execution workflow**: Thiết lập quy trình mẫu cho việc tối ưu hóa bằng GA, LLM-GA.
- **Results display**: In kết quả phân tích giải pháp.
### Configuration and additional files
- **.env**: File cấu hình môi trường, chứa các biến môi trường như API keys.
- **README.md**: Tài liệu hướng dẫn sử dụng và mô tả hệ thống.
- **requirements.txt**: Danh sách các thư viện Python cần thiết.
## Getting Started
### Yêu cầu hệ thống
- **Python**: Version 3.11 or higher
- **IDE**: VSCode (recommended) or any Python IDE
- **Google AI API**: Access to Gemini models
### Download and Install
**Clone/Download the Project**
```bash
git clone https://github.com/giggle-scripter/ads-distribution-ga-llm.git
```
### Thiết lập môi trường
#### 1. **Cài đặt Python 3.13 trở lên**
```bash
python --version
```
Nếu chưa cài đặt, bạn có thể tải xuống từ [python.org](https://www.python.org/downloads/).
#### 2. **Cài đặt các thư viện cần thiết**
```bash
pip install -r requirements.txt
```
### Thiết lập API Key của Google AI
#### 1. **Lấy API Key**
Đăng ký và lấy API Key từ [Google AI Studio](https://makersuite.google.com/app/apikey)
#### 2. **Cấu hình API Key**
Tạo file `.env` trong thư mục gốc và thêm các biến môi trường cần thiết:
```
GOOGLE_API_KEY=your_api_key_here
```
### Tạo bài toán thử nghiệm
Bạn có thể tạo bài toán thử nghiệm bằng 1 trong các phương thức sau:
#### 1. **Tạo từ file**
Tạo 1 file có cấu trúc tương tự như [sample.txt](sample.txt) và sử dụng lệnh:
```python
problem = read_file('path/to/your/file.txt')
```
Cấu trúc file cần tuân theo định dạng đã mô tả trong `problem.py`.
- Line 1: num_billboards num_slots num_ads
- Line 2: slot assignments (which billboard each slot belongs to)
- Line 3: ad base prices
- Line 4: ad max budgets
- Line 5: number of conflicts
- Next lines: conflict pairs (ad1_id ad2_id)
#### 2. **Đọc từ console**
Đọc từ console bằng cách sử dụng hàm `read_console()` trong `problem.py`:
```python
problem = read_console()
```
Cấu trúc nhập từ console tương tự như file.
#### 3. **Tạo ngẫu nhiên**
Đã cung cấp sẵn hàm `random_generate(num_billboards, num_ads)` để khởi tạo bài toán ngẫu nhiên với số lượng biển quảng cáo và quảng cáo nhất định.
```python
problem = random_generate(num_billboards=5, num_ads=10)
```
### Chạy ứng dụng
#### Chạy mẫu có sẵn
Trong file `test.py`, bạn có thể chạy các ví dụ mẫu đã được định nghĩa sẵn bằng cách import các hàm tương ứng
vào `main.py` và chạy:
```bash
python -u main.py
```
Ví dụ `main.py` có thể bao gồm các hàm như:
```python
from test import test_std_ga, test_llm_ga
test_std_ga() # Test thuật toán di truyền tiêu chuẩn
test_llm_ga() # Test thuật toán di truyền tăng cường LLM
```
Bạn hoàn toàn có thể chỉnh sửa cấu hình bằng cách tìm các `ga_config` và `llm_ga_config` trong `test.py`.
#### Chạy từ đầu
Cần tuân theo các bước sau:
1. Import các module cần thiết:
```python
# Import genetic algorithm implementations
from evo import ga, llm_ga

# Import problem handling utilities
from problem import read_file, random_generate

# Import standard libraries
import random
import os
from dotenv import load_dotenv

import google.generativeai as genai

# Import LLM support components
from llm_support import PromptBuilder, LLMSupporter, SOL_PRO_TEMPLATE
```
2. Tải biến môi trường từ file `.env`:
```python
load_dotenv()
```
3. Thiết lập API và Model LLM:
```python
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Initialize the LLM model
# Using Gemini 2.0 Flash for fast, cost-effective responses
model = genai.GenerativeModel('gemini-2.0-flash')

# Set up LLM integration
prompt_builder = PromptBuilder(SOL_PRO_TEMPLATE)
llm_supporter = LLMSupporter(model, prompt_builder)
```

4. Thiết lập seed cho random (cần thiết cho sự ổn định của các lần chạy khác nhau).
```python
random.seed(42)
```

5. Thiết lập cấu hình cho GA, LLM-GA
```python
ga_config = {
    "num_gen": 500,
    "pop_size": 100,
    "pc": 0.8,
    "pm": 0.1,
    "elite_ratio": 0.1,
}

llm_ga_config = {
    "num_gen": 500,
    "pop_size": 100,
    "pc": 0.8,
    "pm": 0.1,
    "elite_ratio": 0.1,
    "max_no_improvement": 40,
    "max_transform_inds": 8,
    "transform_chosen_policy": 'topk',
    "max_time_transform": 8
}
```
Chi tiết các tham số và ý nghĩa xem ở `evo.py`.
6. Đọc bài toán, có thể dùng 1 trong 3 cách đã nêu.
```python
problem = ...
```
7. Gọi hàm `ga` hoặc `llm_ga` để tìm kiếm lời giải.
```python
best_solution = ga(
    num_gen=ga_config["num_gen"],  # Fewer generations for quick comparison
    pop_size=ga_config["pop_size"],
    problem=problem,
    pc=ga_config["pc"],
    pm=ga_config["pm"],
    elite_ratio=ga_config["elite_ratio"]
)
```
hoặc
```python
best_solution = llm_ga(
    num_gen=llm_ga_config["num_gen"],  # Fewer generations for quick comparison
    pop_size=llm_ga_config["pop_size"],
    problem=problem,
    pc=llm_ga_config["pc"],
    pm=llm_ga_config["pm"],
    elite_ratio=llm_ga_config["elite_ratio"],
    max_no_improvement=llm_ga_config["max_no_improvement"],
    max_transform_inds=llm_ga_config["max_transform_inds"],
    transform_chosen_policy=llm_ga_config["transform_chosen_policy"],
    llm_supporter=llm_supporter,
    max_time_transform=llm_ga_config["max_time_transform"]
)
```
8. In kết quả (tham khảo `test.py`).
## Các lỗi thường gặp
### Gặp Rate-Limit với API của Google.
Lỗi này thường mang mã 429 với thông điệp kiểu như
```json
{
  "error": {
    "code": 429,
    "message": "Quota exceeded for quota metric 'Read requests' and limit 'Read requests per minute' of service ...",
    "status": "RESOURCE_EXHAUSTED"
  }
}
```
Gặp lỗi này, bạn có thể chờ một lúc rồi chạy lại, hoặc thiết lập sleep time giữa các lần gọi bằng cách thêm dòng sau vào file `.env`.
```txt
SLEEP_TIME=10
```
Đặt `SLEEP_TIME` càng lớn thì nguy cơ bị rate-limit exceed càng thấp, nhưng hiệu năng sẽ giảm.
### Lỗi API chưa xác thực
```
Error: google.api_core.exceptions.Unauthenticated: 401
```
- Kiểm tra sự tồn tại của file `.env` và xem đã có trường `GOOGLE_API_KEY` với API key lấy từ Google AI Studio.
### Import error
```
ModuleNotFoundError: No module named 'google.generativeai'
```
Hãy chắc chắn đã cài đặt các thư viện cần thiết trong `requirements.txt`.
### JSON Format 
```
Cannot find JSON object in LLM response
```
- This is usually temporary due to LLM response variability
- The algorithm will continue with standard GA operations