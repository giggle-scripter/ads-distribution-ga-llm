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
Trong dự án này, nhóm đưa ra 4 thuật toán chính để giải bài toán:

### Pure Backtracking (Exact Algorithm)
Thuật toán quay lui thuần túy không có tối ưu hóa cắt nhánh, duyệt toàn bộ không gian nghiệm để tìm ra nghiệm tối ưu toàn cục.

**Đặc điểm chính:**
- **Không gian tìm kiếm**: Duyệt một cách có hệ thống toàn bộ không gian nghiệm có thể, mỗi slot có thể được gán một quảng cáo hoặc để trống (-1)
- **Kiểm tra vi phạm**: Tại mỗi bước gán, thuật toán kiểm tra:
  - Ràng buộc conflict: Không có 2 quảng cáo xung đột được gán cùng một billboard
  - Ràng buộc unique: Mỗi quảng cáo chỉ được sử dụng một lần
  - Ràng buộc ngân sách: Chi phí thực tế không vượt quá ngân sách tối đa (tùy chọn)
- **Không có cắt nhánh**: Khác với Branch and Bound, thuật toán này không sử dụng bất kỳ kỹ thuật cắt nhánh nào, đảm bảo khám phá hoàn toàn không gian tìm kiếm
- **Ứng cử viên**: Tại mỗi slot, thuật toán xem xét:
  - Tùy chọn để slot trống (không gán quảng cáo)
  - Tất cả các quảng cáo chưa được sử dụng và không vi phạm ràng buộc
- **Quay lui**: Khi gặp vi phạm ràng buộc hoặc hoàn thành một nhánh, thuật toán quay lui và thử các khả năng khác

**Ưu điểm**: 
- Đảm bảo tìm được nghiệm tối ưu toàn cục
- Đơn giản trong việc triển khai và hiểu thuật toán
- Không có nguy cơ bỏ sót nghiệm tốt do cắt nhánh sai

**Nhược điểm**: 
- Độ phức tạp tăng theo cấp số nhân
- Chậm hơn đáng kể so với Branch and Bound
- Chỉ phù hợp với bài toán kích thước rất nhỏ

### Branch and Bound (Exact Algorithm)
Thuật toán chính xác sử dụng kỹ thuật quay lui (backtracking) kết hợp với cắt nhánh (branch and bound) để tìm ra nghiệm tối ưu toàn cục.

**Đặc điểm chính:**
- **Không gian tìm kiếm**: Duyệt toàn bộ không gian nghiệm có thể, mỗi slot có thể được gán một quảng cáo hoặc để trống (-1)
- **Kiểm tra vi phạm**: Tại mỗi bước gán, thuật toán kiểm tra:
  - Ràng buộc conflict: Không có 2 quảng cáo xung đột được gán cùng một billboard
  - Ràng buộc unique: Mỗi quảng cáo chỉ được sử dụng một lần
  - Ràng buộc ngân sách: Chi phí thực tế không vượt quá ngân sách tối đa (tùy chọn)
- **Cắt nhánh (Pruning)**: 
  - Tính toán cận trên (upper bound) cho doanh thu có thể đạt được từ các slot chưa được gán
  - Nếu doanh thu hiện tại + cận trên ≤ nghiệm tốt nhất đã tìm được, cắt bỏ nhánh này
  - Cận trên được tính bằng cách sắp xếp các cặp (slot, quảng cáo) theo doanh thu tiềm năng giảm dần
- **Ứng cử viên**: Tại mỗi slot, thuật toán xem xét:
  - Tùy chọn để slot trống (không gán quảng cáo)
  - Tất cả các quảng cáo chưa được sử dụng và không vi phạm ràng buộc

**Ưu điểm**: Đảm bảo tìm được nghiệm tối ưu toàn cục với hiệu suất tốt hơn Pure Backtracking
**Nhược điểm**: Độ phức tạp tăng theo cấp số nhân, chỉ phù hợp với bài toán kích thước nhỏ và trung bình

### Genetic Algorithm (GA)
Thuật toán GA được sử dụng với
- Cá thể: Được mã hóa dưới dạng một mảng sol, trong đó `sol[i]=j` nghĩa là slot i được gán quảng cáo j. Nếu `sol[i]=-1` thì slot i không được gán quảng cáo nào
- Lai ghép: Sử dụng phép lai ghép Uniform Crossover, nghĩa là duyệt mỗi vị trí và xem xác suất để lựa chọn giá trị tại vị trí đó theo cha hay mẹ.
- Đột biến: Chọn 1 vị trí (slot) và unassign (đưa về chưa gán) hoặc assign new.
- Selection: Sử dụng Top-K Selection để chọn cá thể cha mẹ.
- Replacement: Luôn giữ lại một vài cá thể tốt nhất của quần thể cũ.
### LLM-GA
Sử dụng LLM như là một phương pháp để thoát khỏi tối ưu cục bộ. Quy trình tại mỗi vòng generation vẫn giống hệt GA. Tuy nhiên nếu sau một vài thế hệ liên tiếp không cải thiện được lời giải, thế hệ kế tiếp sẽ thực hiện crossover và mutation hoàn toàn bằng LLM trên toàn bộ quần thể.

Các toán tử `llm_crossover` và `llm_mutation` hoàn toàn dựa vào LLM để lai ghép và đột biến, với kỳ vọng giúp thoát khỏi tối ưu cục bộ.

### Co-Evo with Memetic
Ý tưởng chính của thuật toán này là tiến hóa song song 2 quần thể
- Quần thể 1 là quần thể các lời giải tiến hóa như GA thông thường.
- Quần thể 2 là quần thể các heuristic để cải thiện lời giải, tiến hóa bằng cách sử dụng LLM-GP.
- Sau một số vòng, việc chuyển giao tri thức sẽ diễn ra, lúc đó một vài heuristic trong quần thể 2 được lựa chọn để áp dụng lên các cá thể của quần thể 1, qua đó giúp cải thiện lời giải.

Với quần thể thứ 2:
- Cá thể: Mỗi cá thể là một đoạn code python với đầu vào là 1 lời giải và 1 problem, đầu ra là 1 lời giải mới.
- Khởi tạo: Dùng LLM để khởi tạo quần thể ban đầu.
- Lai ghép: Dùng LLM để thực hiện `recombine` hai đoạn code cha mẹ.
- Đột biến: Dùng LLM để thực hiện `rephrase` đoạn code cha mẹ.
- Tần suất tiến hóa: Để giảm chi phí, quần thể heuristic sẽ không tiến hóa thường xuyên.
- Đánh giá: Fitness của mỗi cá thể sẽ được cập nhật ở mỗi vòng tiến hóa của quần thể 1 mà quần thể 2 không tiến hóa. Ở mỗi vòng như vậy `new_fitness` của vòng đó được tính bằng độ cải thiện khi áp dụng heuristic lên toàn bộ lời giải trong quần thể 1. Sau đó cập nhật 
```python
fitness = (1-forget_factor)*fitness + forget_factor*new_fitness
```

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

#### 4. `exact_alg.py` - Exact Algorithms
- **PureBacktrackingSupporter**: Triển khai thuật toán quay lui thuần túy không có cắt nhánh
- **BranchAndBoundSupporter**: Triển khai thuật toán quay lui với tối ưu hóa cắt nhánh
- **Feasibility checking**: Kiểm tra tính khả thi của các phép gán
- **Upper bound estimation**: Ước tính cận trên cho thuật toán Branch and Bound

#### 5. `test.py` - Application Entry Point
- **Configuration**: Thiết lập các tham số thuật toán và cấu hình LLM
- **Execution workflow**: Thiết lập quy trình mẫu cho việc tối ưu hóa bằng GA, LLM-GA.
- **Results display**: In kết quả phân tích giải pháp.

#### 6. `test_generator.py` - Test Case Generation
- **Problem generation**: Tạo các bài toán thử nghiệm với các đặc điểm khác nhau
- **Conflict patterns**: Hỗ trợ tạo các mẫu conflict khác nhau (random, clique, star)
- **Parameter variations**: Điều chỉnh các tham số như phân phối giá, ngân sách, tỷ lệ conflict

### Configuration and additional files
- **.env**: File cấu hình môi trường, chứa các biến môi trường như API keys.
- **README.md**: Tài liệu hướng dẫn sử dụng và mô tả hệ thống.
- **requirements.txt**: Danh sách các thư viện Python cần thiết.

## Test Cases Generation
Hệ thống cung cấp 10 test case được tạo tự động với các đặc điểm khác nhau để đánh giá hiệu suất của các thuật toán:

### 1. Small Random (`test_1_small_random.txt`)
- **Cấu hình**: 2 billboards, 10 ads, conflict rate 15%
- **Mục đích**: Test case nhỏ để kiểm tra tính đúng đắn của thuật toán
- **Đặc điểm**: Ít conflict, phù hợp cho exact algorithms

### 2. Medium Random (`test_2_medium_random.txt`)
- **Cấu hình**: 5 billboards, 20 ads, conflict rate 20%
- **Mục đích**: Test case trung bình với mức độ phức tạp vừa phải
- **Đặc điểm**: Cân bằng giữa kích thước và độ phức tạp

### 3. Large Random (`test_3_large_random.txt`)
- **Cấu hình**: 15 billboards, 50 ads, conflict rate 25%
- **Mục đích**: Test case lớn để đánh giá khả năng mở rộng
- **Đặc điểm**: Thách thức cho exact algorithms, phù hợp cho GA

### 4. Small Clique (`test_4_small_clique.txt`)
- **Cấu hình**: 3 billboards, 16 ads, 2 cliques size ~5
- **Mục đích**: Test pattern conflict dạng clique (tất cả conflict với nhau)
- **Đặc điểm**: Conflict tập trung, tạo ra các ràng buộc mạnh

### 5. Large Clique (`test_5_large_clique.txt`)
- **Cấu hình**: 15 billboards, 50 ads, 4 cliques size ~10
- **Mục đích**: Test clique pattern ở quy mô lớn
- **Đặc điểm**: Nhiều nhóm conflict độc lập

### 6. Medium Star (`test_6_medium_star.txt`)
- **Cấu hình**: 7 billboards, 30 ads, 5 centroids với 5 satellites mỗi cái
- **Mục đích**: Test pattern conflict dạng sao (một ad conflict với nhiều ad khác)
- **Đặc điểm**: Ads có base price cao trở thành trung tâm conflict

### 7. Medium Robust (`test_7_medium_robust.txt`)
- **Cấu hình**: 7 billboards, 30 ads, conflict rate 30%, robust price distribution
- **Mục đích**: Test với phân phối giá không đồng đều (20% ads có giá cao 500, 80% ads có giá 200-300)
- **Đặc điểm**: Tạo ra sự khác biệt lớn về giá trị ads

### 8. Medium Approx Budget (`test_8_medium_approx_budget.txt`)
- **Cấu hình**: 7 billboards, 30 ads, conflict rate 30%, max_budget ≈ base_price
- **Mục đích**: Test khi max budget gần bằng base price (budget constraint chặt)
- **Đặc điểm**: Ràng buộc ngân sách ảnh hưởng mạnh đến doanh thu

### 9. Medium Low Budget (`test_9_medium_low_budget.txt`)
- **Cấu hình**: 7 billboards, 30 ads, conflict rate 30%, max_budget < base_price
- **Mục đích**: Test khi max budget thấp hơn base price
- **Đặc điểm**: Revenue bị giới hạn mạnh bởi budget constraint

### 10. Many Ads (`test_10_many_ads.txt`)
- **Cấu hình**: 8 billboards, 70 ads, conflict rate 30%
- **Mục đích**: Test với số lượng ads lớn so với số slots
- **Đặc điểm**: Nhiều lựa chọn, cạnh tranh cao giữa các ads

### Các tham số sinh test:
- **base_price_dist**: 
  - `'uniform'`: Phân phối đều từ 200-300
  - `'robust'`: 20% ads có giá 500, 80% có giá 200-300
- **max_budget_type**:
  - `'high'`: max_budget = base_price × (1.3-2.5)
  - `'approx'`: max_budget = base_price × (1.05-2.05)  
  - `'low'`: max_budget = base_price × (1.05-1.55)
- **conflict_rate**: Tỷ lệ conflict giữa các cặp ads (0.0-1.0)

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
#### 4. **Sử dụng test cases có sẵn**
Sử dụng các test case đã được tạo sẵn trong thư mục `test/`:
```python
problem = read_file('test/test_1_small_random.txt')
```
#### 5. **Tạo test cases tùy chỉnh**
Sử dụng các hàm trong `test_generator.py`:
```python
from test_generator import generate_randomly_conflict, generate_clique_conflict, generate_star_conflict

# Tạo problem với random conflicts
problem = generate_randomly_conflict(num_billboards=5, num_ads=20, conflict_rate=0.3)

# Tạo problem với clique conflicts  
problem = generate_clique_conflict(num_billboards=3, num_ads=16, clique_size=5, max_num_cliques=2)

# Tạo problem với star conflicts
problem = generate_star_conflict(num_billboards=7, num_ads=30, num_centroids=5, satellites=5)
```

### Chạy ứng dụng
#### Chạy mẫu có sẵn
Trong file `auto_test.py`, bạn có thể chạy các ví dụ mẫu đã được định nghĩa sẵn bằng cách import các hàm tương ứng vào `main.py` và chạy:
```bash
python -u main.py
```
Ví dụ `main.py` có thể bao gồm các hàm như:
```python
from auto_test import test_std_ga, test_llm_ga
test_std_ga() # Test thuật toán di truyền tiêu chuẩn
test_llm_ga() # Test thuật toán di truyền tăng cường LLM
```
Bạn hoàn toàn có thể chỉnh sửa cấu hình và chạy lại.
#### Chạy từ đầu
Cần tuân theo các bước sau:
1. Import các module cần thiết:
```python
import os
from dotenv import load_dotenv
import google.generativeai as genai

import random

from problem import read_file, read_console, Problem
from llm_support import LLMSupporter, PromptBuilder
from exact_alg import branch_and_bound, pure_backtracking
from evo import ga, llm_ga
from co_evo import co_evo_llm
from heuristic import hill_climbing

# Load enviroment variables
load_dotenv()
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
prompt_builder = PromptBuilder()
llm_supporter = LLMSupporter(model)
```

4. Thiết lập seed cho random (cần thiết cho sự ổn định của các lần chạy khác nhau).
```python
random.seed(42)
```

5. Đọc problem từ file hoặc từ console
```python
# Đọc từ console
problem = read_console()

# Đọc từ file
problem = read_file('/path/to/test')
```

6. Thực hiện gọi hàm các thuật toán
- Exact Algorithm
```python
# Giải bằng Pure backtracking
sol, stats = pure_backtracking(problem, time_limit=2000.0)
print(sol)
print(stats) # Thống kê việc giải

# Giải bằng branch and bound
sol, stats = branch_and_bound(problem, time_limit=2000.0)
print(sol)
print(stats) # Thống kê việc giải
```
- Các thuật toán tiến hóa
```python
# Giải bằng GA
best, stats = ga(num_gen=800, pop_size=100, problem=problem,
                 pc=0.8, pm=0.2, elite_ratio=0.1,
                 debug=False)
print(best.chromosome)
print(stats)

# LLM - GA
best, stats = llm_ga(num_gen=800, pop_size=100, problem=problem,
                     pc=0.8, pm=0.2, elite_ratio=0.1,
                     llm_supporter=llm_supporter,
                     prompt_builder=prompt_builder,
                     max_no_improve=90, max_llm_call=8,
                     debug=False)
print(best.chromosome)
print(stats)

# Co-Evo-Memetic
best, stats = co_evo_llm(800, 100, 16, problem, llm_supporter,
                         prompt_builder, 
                         pc=0.8, pm=0.2, elite_ratio=0.1,
                         heuristic_evo_cycle=80, apply_heuristic_cycle=80,
                         early_stopping_gen=600, appliable_heuristics=4,
                         problem_code_filepath='safety_problem_code.txt',
                         debug=False)
```

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