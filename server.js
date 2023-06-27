const express = require('express')
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

const app = express()
const port = 3000

var storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, 'uploads/')
    },
    filename: function (req, file, cb) {  
        cb(null, file.originalname)
    }
});

const upload = multer({ storage: storage });

app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static("public"));

app.post('/predict_all_models', upload.single('video'), async (req, res) => {
    const moviePredictionAPI = 'http://127.0.0.1:8000/predict_sequence/all_models'
    const fileStream = fs.createReadStream(req.file.path);
    const formData = new FormData();
    formData.append('video', fileStream);

    try {
        const response = await axios.post(moviePredictionAPI, formData, {
            headers: {
                'Content-Type': `multipart/form-data; boundary=${formData._boundary}`,
            }
        });
        fs.unlink(req.file.path, (err) => {
            if (err) {
                console.error(err);
                return;
            }
            console.log(req.file.path + ' was deleted');
        });

        res.status(200).json(response.data);
    } catch (error) {
        console.log(error);
        res.status(500).send("An error occurred while processing the video.");
    }
});

app.post('/predict_custom_model_3d_cnn', upload.single('video'), async (req, res) => {
    const moviePredictionAPI = 'http://127.0.0.1:8000/predict_sequence/custom_model_3d_cnn'
    const fileStream = fs.createReadStream(req.file.path);
    const formData = new FormData();
    formData.append('video', fileStream);

    try {
        const response = await axios.post(moviePredictionAPI, formData, {
            headers: {
                'Content-Type': `multipart/form-data; boundary=${formData._boundary}`,
            }
        });
        fs.unlink(req.file.path, (err) => {
            if (err) {
                console.error(err);
                return;
            }
            console.log(req.file.path + ' was deleted');
        });

        res.status(200).json(response.data);
    } catch (error) {
        console.log(error);
        res.status(500).send("An error occurred while processing the video.");
    }
});

app.post('/predict_custom_model_2d_cnn', upload.single('video'), async (req, res) => {
    const moviePredictionAPI = 'http://127.0.0.1:8000/predict_sequence/custom_model_2d_cnn'
    const fileStream = fs.createReadStream(req.file.path);
    const formData = new FormData();
    formData.append('video', fileStream);

    try {
        const response = await axios.post(moviePredictionAPI, formData, {
            headers: {
                'Content-Type': `multipart/form-data; boundary=${formData._boundary}`,
            }
        });
        fs.unlink(req.file.path, (err) => {
            if (err) {
                console.error(err);
                return;
            }
            console.log(req.file.path + ' was deleted');
        });

        res.status(200).json(response.data);
    } catch (error) {
        console.log(error);
        res.status(500).send("An error occurred while processing the video.");
    }
});

app.post('/predict_inceptionv3', upload.single('video'), async (req, res) => {
    const moviePredictionAPI = 'http://127.0.0.1:8000/predict_sequence/inceptionv3'
    const fileStream = fs.createReadStream(req.file.path);
    const formData = new FormData();
    formData.append('video', fileStream);

    try {
        const response = await axios.post(moviePredictionAPI, formData, {
            headers: {
                'Content-Type': `multipart/form-data; boundary=${formData._boundary}`,
            }
        });
        fs.unlink(req.file.path, (err) => {
            if (err) {
                console.error(err);
                return;
            }
            console.log(req.file.path + ' was deleted');
        });

        res.status(200).json(response.data);
    } catch (error) {
        console.log(error);
        res.status(500).send("An error occurred while processing the video.");
    }
});

app.post('/predict_resnet50', upload.single('video'), async (req, res) => {
    const moviePredictionAPI = 'http://127.0.0.1:8000/predict_sequence/resnet50'
    const fileStream = fs.createReadStream(req.file.path);
    const formData = new FormData();
    formData.append('video', fileStream);

    try {
        const response = await axios.post(moviePredictionAPI, formData, {
            headers: {
                'Content-Type': `multipart/form-data; boundary=${formData._boundary}`,
            }
        });
        fs.unlink(req.file.path, (err) => {
            if (err) {
                console.error(err);
                return;
            }
            console.log(req.file.path + ' was deleted');
        });

        res.status(200).json(response.data);
    } catch (error) {
        console.log(error);
        res.status(500).send("An error occurred while processing the video.");
    }
});

app.post('/predict_vgg16_model_1', upload.single('video'), async (req, res) => {
    const moviePredictionAPI = 'http://127.0.0.1:8000/predict_sequence/vgg16_model_1'
    const fileStream = fs.createReadStream(req.file.path);
    const formData = new FormData();
    formData.append('video', fileStream);

    try {
        const response = await axios.post(moviePredictionAPI, formData, {
            headers: {
                'Content-Type': `multipart/form-data; boundary=${formData._boundary}`,
            }
        });
        fs.unlink(req.file.path, (err) => {
            if (err) {
                console.error(err);
                return;
            }
            console.log(req.file.path + ' was deleted');
        });

        res.status(200).json(response.data);
    } catch (error) {
        console.log(error);
        res.status(500).send("An error occurred while processing the video.");
    }
});

app.post('/predict_vgg16_model_2', upload.single('video'), async (req, res) => {
    const moviePredictionAPI = 'http://127.0.0.1:8000/predict_sequence/vgg16_model_2'
    const fileStream = fs.createReadStream(req.file.path);
    const formData = new FormData();
    formData.append('video', fileStream);

    try {
        const response = await axios.post(moviePredictionAPI, formData, {
            headers: {
                'Content-Type': `multipart/form-data; boundary=${formData._boundary}`,
            }
        });
        fs.unlink(req.file.path, (err) => {
            if (err) {
                console.error(err);
                return;
            }
            console.log(req.file.path + ' was deleted');
        });
        
        res.status(200).json(response.data);
    } catch (error) {
        console.log(error);
        res.status(500).send("An error occurred while processing the video.");
    }
});

app.listen(port, () => console.log(`Example app listening on port ${port}!`))