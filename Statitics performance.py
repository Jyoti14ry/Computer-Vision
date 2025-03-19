import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sample DataFrame (or read from CSV)
data = {
    'Name': ['Ram', 'Sita', 'Gita', 'Shyam', 'Jyoti'],
    'Math': [78, 85, 90, np.nan, 88],
    'Science': [82, 79, np.nan, 91, 85],
    'English': [88, 92, 87, 85, np.nan]
}

df = pd.DataFrame(data)

# Fill missing values with the column mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# Calculate total marks and average
df['Total Marks'] = df[['Math', 'Science', 'English']].sum(axis=1)
df['Average'] = df[['Math', 'Science', 'English']].mean(axis=1)

# Assign Grades
def assign_grade(avg):
    if avg >= 90:
        return 'A'
    elif avg >= 80:
        return 'B'
    elif avg >= 70:
        return 'C'
    else:
        return 'D'

df['Grade'] = df['Average'].apply(assign_grade)

# Display final DataFrame
print(df)

# Visualization: Bar Chart of Student Performance
plt.figure(figsize=(8, 5))
plt.bar(df['Name'], df['Total Marks'], color=['blue', 'green', 'red', 'purple', 'orange'])
plt.xlabel('Students')
plt.ylabel('Total Marks')
plt.title('Student Performance')
plt.show()
