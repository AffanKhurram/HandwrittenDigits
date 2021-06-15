'use strict';

// Load stuff for drawing
var canvas = document.getElementById('canvas');
var placeAnswer = document.getElementById('value');
var sigpad = new SignaturePad(canvas, {
    backgroundColor: 'rgb(255, 255, 255)',
    minWidth: 8,
    maxWidth: 8
});

// Load the buttons
var save = document.getElementById('save');
var clear = document.getElementById('clear');

// Load the model
var model;
tf.loadLayersModel('tfjs_model/model.json')
    .then((res) => { model = res; })
    .catch((err) => { console.log(err); });


// Click functions for the buttons
save.addEventListener('click', function(event) {
    console.log('original img', sigpad.toDataURL());


    placeAnswer.innerHTML = "You wrote a(n): Working...";

    var img = process(canvas);
    var pred = model.predict(img.reshape([1, 28, 28, 1])).argMax(1).bufferSync().get(0);
    placeAnswer.innerHTML = "You wrote a(n): " + String(pred);
});

clear.addEventListener('click', function(event) {
    sigpad.clear()
});
