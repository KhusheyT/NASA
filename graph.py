import matplotlib.pyplot as plt

# Example data (replace with your actual data)
# Dates or time intervals (x-axis)
dates = ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']
# Corresponding damage areas (y-axis)
damage_areas = [10, 15, 18, 12, 10]
# Corresponding number of fire incidents (y-axis)
fire_incidents = [5, 7, 8, 6, 5]

# Create a figure and axis for the plot
fig, ax1 = plt.subplots()

# Plot the damage area data
ax1.set_xlabel('Date')
ax1.set_ylabel('Damage Area (sq km)', color='tab:blue')
ax1.plot(dates, damage_areas, color='tab:blue', marker='o', label='Damage Area')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_xticklabels(dates, rotation=45)  # Rotate x-axis labels for readability

# Create a second y-axis for fire incidents
ax2 = ax1.twinx()
ax2.set_ylabel('Fire Incidents', color='tab:red')
ax2.plot(dates, fire_incidents, color='tab:red', marker='x', label='Fire Incidents')
ax2.tick_params(axis='y', labelcolor='tab:red')

# Add legends for both datasets
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right')

# Set a title and display the plot
plt.title('Damage Area vs. Fire Incidents Over Time')
plt.tight_layout()
plt.show()
