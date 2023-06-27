const videoInput = document.getElementById('video');
const actionButtons = document.querySelectorAll('.action-btn');
const responseList = document.querySelector('.response-list');
const loadingAnimation = document.querySelector('.loading-animation');
const form = document.querySelector('form');

const endpoints = [
    'http://localhost:3000/predict_all_models',
    'http://localhost:3000/predict_custom_model_3d_cnn',
    'http://localhost:3000/predict_custom_model_2d_cnn',
    'http://localhost:3000/predict_inceptionv3',
    'http://localhost:3000/predict_resnet50',
    'http://localhost:3000/predict_vgg16_model_1',
    'http://localhost:3000/predict_vgg16_model_2',
];

videoInput.addEventListener('change', (event) => {
    const hasFile = event.target.files.length > 0;
    const fileLabel = document.getElementById('file-label');

    if (hasFile) {
        fileLabel.textContent = event.target.files[0].name;
    } else {
        fileLabel.textContent = 'Choose a video sequence';
    }

    actionButtons.forEach((button) => {
        button.disabled = !hasFile;
    });
});

function showToast() {
    const toast = document.querySelector('.toast');
    toast.hidden = false;
    toast.classList.add('show');
  
    setTimeout(() => {
      toast.classList.remove('show');
      toast.hidden = true;
    }, 5000);
  }

actionButtons.forEach((button, index) => {
    button.addEventListener('click', async (event) => {
        event.preventDefault();

        const formData = new FormData();
        formData.append('video', videoInput.files[0]);

        actionButtons.forEach((button) => {
            button.disabled = true;
        });

        responseList.innerHTML = '';
        loadingAnimation.hidden = false;

        const response = await fetch(endpoints[index], {
            method: 'POST',
            body: formData,
        });

        const responseData = await response.json();

        showToast();
        loadingAnimation.hidden = true;

        responseList.innerHTML = '';
        Object.entries(responseData).forEach(([key, value]) => {
            const listItem = document.createElement('li');
            listItem.innerHTML = `${key}: <b>${value}</b>`;
            responseList.appendChild(listItem);
        });

        actionButtons.forEach((button) => {
            button.disabled = false;
        });
    });
});
