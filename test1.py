import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import os

#for individual format
save_folder = r'..\Content-Major'
# os.makedirs(save_folder, exist_ok=True)
for j in range(1,3):
    #for house wise format
    # save_folder = r'..\Content-Major\House-wise\House_'+str(j)
    # os.makedirs(save_folder, exist_ok=True)
    file = pd.read_csv(f'house{j}.csv')
    cd = file
    cd['localminute'] = pd.to_datetime(cd['localminute'])

    for i in range(0,1):
        if i<=1:
            df = cd[0:24*(10**i)]
            days = 1*(10**i)
        else:
            df = cd[0:24*(30)]
            days = 30
        plt.figure(figsize=(12, 6))

        #for savin in day-wise format 
        # save_folder = r'..\Content-Major\Days-wise\for_'+str(days)+'_days'
        # os.makedirs(save_folder, exist_ok=True)
        
        plt.plot(df['localminute'], df['use'], label=f'Energy Consumption - {days} day{"s" if days > 1 else ""} ')
        plt.plot(df['localminute'], df['gen'], label=f'Energy Generation - {days} day{"s" if days > 1 else ""}')
        plt.title(f'Energy Trends of House {j} - for {days} day{"s" if days > 1 else ""}', fontsize=12)
        
        ax = plt.gca()
        if days <= 1:
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))  # Every 3 hours
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # Only time
        elif days <= 10:
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))  # Every 6 hours
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b %H:%M'))  # Date & Time
        elif days <= 30:
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))  # Every 2 days
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
        else:
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))  # Every 5 days
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))

        # Rotate labels, add grid
        plt.gcf().autofmt_xdate(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.5)

        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Energy (kWh)')
        # plt.show()

        
        filename = f"House_{j}_Energy_{days}_days.png"
        filepath = os.path.join(save_folder, filename)

        
        plt.savefig(filepath, dpi=300)  # dpi=300 for high resolution
        plt.close()
        



# normalized_columns = ['use', 'gen', 'grid', 'DHI', 'DNI', 'GHI']
# scaler = MinMaxScaler()

# df[normalized_columns] = scaler.fit_transform(df[normalized_columns])

# df[normalized_columns].to_csv('normalized_house_1.csv', index=False)

# plt.figure(figsize=(12, 6))
# plt.plot(df['localminute'], df['use'], label='Energy Consumption')
# plt.plot(df['localminute'], df['gen'], label='Energy Generation')
# plt.legend()
# plt.xlabel('Time')
# plt.ylabel('Energy (kWh)')
# plt.title('Energy Trends')
# plt.show()