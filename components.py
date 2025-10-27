import matplotlib.pyplot as plt

# Data for the pie chart
sizes = [68,0.001,27,5]
labels = ['Dark Energy', 'Radiation', 'Dark Matter', 'Baryonic Matter']

# Create the pie chart
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90  )

# Ensure the pie chart is drawn as a circle
plt.axis('equal')

# Display the pie chart
plt.title('Components of the Universe')
plt.show()