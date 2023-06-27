import matplotlib.pyplot as plt

train_accuracy = 54.73
validation_accuracy = 51.30
test_accuracy = 48.58

labels = ['Antrenare', 'Validare', 'Testare']
values = [train_accuracy, validation_accuracy, test_accuracy]
colors = ['green', 'blue', 'orange']

plt.barh(labels, values, color=colors)
plt.xlabel('Acuratețea')
plt.title('Acuratețea la antrenare, testare și validare')

plt.show()
