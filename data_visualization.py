"""import pandas as pd
import matplotlib.pyplot as plt

def main():
    
    #create a visualization of the activity the person worked on 
    #in a large distribution from left to right with less to more hours 
    #worked, and the person's name underneath
    #this will be horizontal bar chart

    #this is what i have for now, and I'll be creating a grouped bar chart 
    #for the next plot I'll have so I can sort things in ascending order 
    #to get the distribution I want per activity.

    file_path = 'B:\\Realization Project\\Medical WIP 2023\\Project Hours Pivot Table.xlsx'
    df = pd.read_excel(file_path, header=2)

    #group by 'Name' and 'Activity', then sum 'Hours', and finally reset index to flatten 
    #dataframe
    grouped = df.groupby(['Name', 'Activity'])['Hours'].sum().reset_index()

    #calculate total hours worked by each person 
    total_hours = df.groupby('Name')['Hours'].sum().reset_index() 
    total_hours.rename(columns={'Hours': 'Total Hours'}, inplace=True)

    #merge total hours back into grouped DataFrame 
    merged_df = pd.merge(grouped, total_hours, on='Name')

    #get unique names and activities for plotting
    unique_names = merged_df['Name'].unique()
    unique_activities = merged_df['Activity'].unique()

    print("the unique activities: " + str(unique_activities))

    #Plotting
    fig, ax = plt.subplots(figsize=(12, 8))

    #set positions and width for bars
    width = 0.35
    positions = range(len(unique_names))

    #plot bars for each activity
    for i, activity in enumerate(unique_activities):
        if activity == 'Default Task':
            activity_hours = merged_df[merged_df['Activity'] == activity]['Hours'].reindex(unique_names).fillna(0)
            ax.bar([p + i * width for p in positions], activity_hours, width, label=activity)

    #now, sort dataframe by 'Hours' in ascending order to get distribution
    ax.bar([p + 1 * width for p in positions], 
           total_hours.set_index('Name')['Total Hours'].reindex(unique_names).fillna(0), width, label='Total Hours')
    
    #set labels for x ticks
    ax.set_xticks([p + width * 1 / 2 for p in positions])
    ax.set_xticklabels(unique_names)
    #had to set tick labels for unique names instead

    #adding legend and title
    ax.legend()
    ax.set_xlabel('Person')
    ax.set_ylabel('Hours Worked')
    ax.set_title('Hours Worked on Specific Activities vs. Total Hours Worked')

    #show plot
    plt.show()


main()"""




#modified code- need to upload to repository
import pandas as pd
import matplotlib.pyplot as plt

def main():
    
    #create a visualization of the activity the person worked on 
    #in a large distribution from left to right with less to more hours 
    #worked, and the person's name underneath
    #this will be horizontal bar chart

    #this is what i have for now, and I'll be creating a grouped bar chart 
    #for the next plot I'll have so I can sort things in ascending order 
    #to get the distribution I want per activity.


    file_path = 'B:\\Realization Project\\Medical WIP 2023\\Project Hours Pivot Table.xlsx'
    df = pd.read_excel(file_path, header=2)

    #group by 'Name' and 'Activity', then sum 'Hours', and finally reset index to flatten 
    #dataframe
    grouped = df.groupby(['Name', 'Activity'])['Hours'].sum().reset_index()


    #filter dataframe for 'default task' activity
    default_task_df = grouped[grouped['Activity'] == 'Default Task']
    print("the default task dataframe: " + str(default_task_df))

    #plot the default task activity 
    fig, ax = plt.subplots(figsize=(12, 8))

    #plot the bar plots for default task activity 
    ax.barh(default_task_df['Name'], default_task_df['Hours'], label='Default Task')

    #adding labels and title
    ax.set_xlabel('Hours Worked')
    ax.set_ylabel('Person')
    ax.set_title('Hours Worked on default task by each person')

    #adding the legend
    ax.legend()

    #show the plot 
    plt.show()

main()


#next step is to narrow down by specific audits worked on, and info. about the default task itself
#error coming because matplotlib not updated for ax.barh



#mismatch in lengths of data trying to plot 
#positions list and merged_df['Hours'] series need to have same length 
#for bar plot to work properly


"""import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

#load and preprocess data 
df = pd.read_csv('billable_hours.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Month'] = df['Date'].dt.month

#feature engineering
X = df[['DayOfWeek', 'Month']]
y = df['Billable']
//working on stuff
#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#model training
model = LinearRegression()
model.fit(X_train, y_train)

#predictions
y_pred = model.predict(X_test)

#evaluation
print("Mean absolute error:", mean_absolute_error(y_test, y_pred))

#visualization
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Billable Hours'], label='Actual Billable Hours')
plt.plot(df['Date'].iloc[len(y_train):], y_pred, label='Predicted Billable Hours', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Billable Hours')
plt.title('Billable Hours Prediction')
plt.legend()
plt.show()"""
