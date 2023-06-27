import matplotlib.pyplot as plt

train_accuracy = 71.43
validation_accuracy = 85.36
test_accuracy = 84.96

labels = ['Antrenare', 'Validare', 'Testare']
values = [train_accuracy, validation_accuracy, test_accuracy]
colors = ['green', 'blue', 'orange']

plt.barh(labels, values, color=colors)
plt.xlabel('Acuratețea')
plt.title('Acuratețea la antrenare, testare și validare')

plt.show()
