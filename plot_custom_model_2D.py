import matplotlib.pyplot as plt

train_accuracy = 99.09
validation_accuracy = 98.75
test_accuracy = 87.91

labels = ['Antrenare', 'Validare', 'Testare']
values = [train_accuracy, validation_accuracy, test_accuracy]
colors = ['green', 'blue', 'orange']

plt.barh(labels, values, color=colors)
plt.xlabel('Acuratețea')
plt.title('Acuratețea la antrenare, testare și validare')

plt.show()
