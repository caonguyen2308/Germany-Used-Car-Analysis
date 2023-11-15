#1: Vẽ bar chart các loại xe mec đã qua sử dụng

import matplotlib.pyplot as plt
import numpy as np
import csv
from mpl_toolkits.mplot3d import Axes3D

file_path = 'D:/Cybersoft/Mid Test/merc.csv'
row_list = []
with open(file_path, newline='', encoding='utf-8') as csv_file:
    data_csv = csv.reader(csv_file, delimiter=',', skipinitialspace=True, quotechar='"')
    header = next(data_csv)
    print(header)
    for row in data_csv:
        row_list.append(row)

# Xóa thư viện đếm của các loại xe
model_counts = {}

for row in row_list:
    model = row[0]  
    if model in model_counts:
        model_counts[model] += 1
    else:
        model_counts[model] = 1


model_names = list(model_counts.keys())
model_count_values = list(model_counts.values())

plt.figure(figsize= (12, 8) )
plt.bar(model_names, model_count_values)
plt.xlabel('Loại xe')
plt.ylabel('Số lượng')
plt.title('Số lượng xe Mercedes đã qua sử dụng')
plt.xticks(rotation=45)
#plt.show()


#2: Vẽ pie chart tính tỉ lệ các dòng xe Mec trang bị cần số và nhiên liệu sử dụng

transmission_count = {}
fuelType_count = {}

# Đếm số lượng lẫy chuyển số
for row in row_list:
    transmission = row[3]
    if transmission in transmission_count:
        transmission_count[transmission] += 1
    else:
        transmission_count[transmission] = 1

# Đếm số lượng loại nhiên liệu
for row in row_list:
    fuelType = row[5]
    if fuelType in fuelType_count:
        fuelType_count[fuelType] += 1
    else:
        fuelType_count[fuelType] = 1

plt.figure(figsize= (12, 6) )
# Vẽ pie lẫy chuyển số
plt.subplot(1, 2, 1)
plt.pie(transmission_count.values(), labels=transmission_count.keys(), autopct='%.2f%%')
plt.title('Các loại lẫy chuyển số của xe')
# Vẽ pie loại nhiên liệu
plt.subplot(1, 2, 2)
plt.pie(fuelType_count.values(), labels=fuelType_count.keys(), autopct='%.2f%%')
plt.title('Các loại nhiên liệu')

plt.tight_layout()  # Canh giữa 2 pie
#plt.show()

#3 Vẽ biểu đồ line chart so sánh giữa giá xe và mileage
yearwise_price = {}
yearwise_mileage = {}


for row in row_list:
    year = row[1]
    price = float(row[2])  # Convert price to float
    mileage = float(row[4])  # Convert mileage to float
    
    if year not in yearwise_price:
        yearwise_price[year] = []
        yearwise_mileage[year] = []
    
    yearwise_price[year].append(price)
    yearwise_mileage[year].append(mileage)

avg_price_by_year = {}
avg_mileage_by_year = {}

for year, prices in yearwise_price.items():
    avg_price_by_year[year] = sum(prices) / len(prices)

for year, mileages in yearwise_mileage.items():
    avg_mileage_by_year[year] = sum(mileages) / len(mileages)


years = list(avg_price_by_year.keys())
avg_prices = list(avg_price_by_year.values())
avg_mileages = list(avg_mileage_by_year.values())

# Reverse the order of years to arrange from past to present
years.reverse()

plt.figure(figsize=(10, 6))
plt.plot(years, [avg_price_by_year[year] for year in years], label='Average Price')
plt.plot(years, [avg_mileage_by_year[year] for year in years], label='Average Mileage')


plt.xlabel('Năm')
plt.ylabel('Giá trị')
plt.title('Giá trung bình và số kilomet theo từng năm')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
#plt.show()



#4: Vẽ hình plot về mileage and year

#5: Vẽ 2D và 3D scatter về EngineSize, Tax và MPH

# Scatter 3D
engineSize_counts = {1.0: 5, 1.5: 8, 2.0: 10, 2.5: 6}
Tax_counts = {100: 7, 150: 9, 200: 12, 250: 5}
Mph_counts = {120: 8, 130: 10, 140: 9, 150: 6}

# Xác định loại dữ liệu và đếm
engineSize_categories, engineSize_data = zip(*sorted(engineSize_counts.items()))
Tax_categories, Tax_data = zip(*sorted(Tax_counts.items()))
Mph_categories, Mph_data = zip(*sorted(Mph_counts.items()))

# Tạo figure 3D
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Tạo set x-value
x_values = np.array(engineSize_categories)

# Scatter plot for EngineSize with larger markers (s parameter)
ax.scatter(x_values, np.zeros_like(x_values), engineSize_data, label='EngineSize', c='r', marker='o', s=100)

# Scatter plot for Tax with larger markers
ax.scatter(x_values, np.zeros_like(x_values), Tax_data, label='Tax', c='g', marker='s', s=100)

# Scatter plot for Mph with larger markers
ax.scatter(x_values, np.zeros_like(x_values), Mph_data, label='Mph', c='b', marker='^', s=100)


ax.set_xlabel('Giá trị x')
ax.set_ylabel('Giá trị y')
ax.set_zlabel('Giá trị')
ax.set_title('Biểu đồ sự phân bố của kích thước động cơ, giá mua và chỉ số tốc độ')
ax.legend(loc='upper right')
plt.show()

# Scatter 2D
engineSize_counts = {1.0: 5, 1.5: 8, 2.0: 10, 2.5: 6}
Tax_counts = {100: 7, 150: 9, 200: 12, 250: 5}
Mph_counts = {120: 8, 130: 10, 140: 9, 150: 6}

# Xác định loại dữ liệu và đếm
engineSize_categories, engineSize_data = zip(*sorted(engineSize_counts.items()))
Tax_categories, Tax_data = zip(*sorted(Tax_counts.items()))
Mph_categories, Mph_data = zip(*sorted(Mph_counts.items()))

# Tạo set x-value
x_values = np.array(engineSize_categories)

# Tạo figure
fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(x_values, engineSize_data, label='EngineSize', c='r', marker='o', s=100)
ax.scatter(x_values, Tax_data, label='Tax', c='g', marker='s', s=100)
ax.scatter(x_values, Mph_data, label='Mph', c='b', marker='^', s=100)

ax.set_xlabel('Loại')
ax.set_ylabel('Giá trị')
ax.set_title('Biểu đồ sự phân bố của kích thước động cơ, giá mua và chỉ số tốc độ')
ax.legend(loc='upper right')
plt.show()














        



