import matplotlib.pyplot as plt

train_accuracy = 61.39
validation_accuracy = 66.95
test_accuracy = 40.10

labels = ['Antrenare', 'Validare', 'Testare']
values = [train_accuracy, validation_accuracy, test_accuracy]
colors = ['green', 'blue', 'orange']

plt.barh(labels, values, color=colors)
plt.xlabel('Acuratețea')
plt.title('Acuratețea la antrenare, testare și validare')

plt.show()
