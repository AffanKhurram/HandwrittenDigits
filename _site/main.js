'use strict';

import { Spinner } from "./modules/spin.js";

var canvas = document.getElementById('canvas');
var placeAnswer = document.getElementById('value');
var sigpad = new SignaturePad(canvas, {
    backgroundColor: 'rgb(255, 255, 255)',
    minWidth: 8,
    maxWidth: 8
});

var save = document.getElementById('save');
var clear = document.getElementById('clear');

var opts = {
    lines: 12, // The number of lines to draw
    length: 0, // The length of each line
    width: 10, // The line thickness
    radius: 18, // The radius of the inner circle
    scale: 1, // Scales overall size of the spinner
    corners: 1, // Corner roundness (0..1)
    speed: 1, // Rounds per second
    rotate: 7, // The rotation offset
    animation: 'spinner-line-fade-quick', // The CSS animation name for the lines
    direction: 1, // 1: clockwise, -1: counterclockwise
    color: '#ffffff', // CSS color or array of colors
    fadeColor: 'transparent', // CSS color or array of colors
    top: '50%', // Top position relative to parent
    left: '50%', // Left position relative to parent
    shadow: '0 0 1px transparent', // Box-shadow for the lines
    zIndex: 2000000000, // The z-index (defaults to 2e9)
    className: 'spinner', // The CSS class to assign to the spinner
    position: 'absolute', // Element positioning
};
var spinner = new Spinner(opts);

save.addEventListener('click', function(event) {
    console.log('original img', sigpad.toDataURL());
    spinner.spin(placeAnswer);
    var tensor = tf.browser.fromPixels(canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height)).mean(2);
    var small = resize(tensor);
    var normalized = invert(small);
    var center = getCenter(normalized);

    var before_x = 14 - center.x;
    var before_y = 14 - center.y;
    var after_x = 28 - (before_x + normalized.shape[1]);
    var after_y = 28 - (before_y + normalized.shape[0]);

    var finalImage = normalized.pad([[before_y, after_y], [before_x, after_x]]);

    var model = tf.loadLayersModel('tfjs_model/model.json');
    model.then(function(res) {
        var pred = res.predict(finalImage.reshape([1, 28, 28, 1])).argMax(1).bufferSync().get(0);
        spinner.stop();
        placeAnswer.innerHTML = "You wrote a(n): " + String(pred);
    });
});



clear.addEventListener('click', function(event) {
    sigpad.clear()
});

// crops and resizes img down to 20 by 20 while keeping the aspect ratio
function resize(img) {
    var small = img.bufferSync();
    var startr = small.shape[1];
    var endr = 0;
    var startc = small.shape[0];
    var endc = 0;
    for (var i = 0; i < small.shape[0]; i++) {
        for (var j = 0; j < small.shape[1]; j++) {

            // If very dark square
            if (small.get(i, j) < 50) {
                if (i < startr) {
                    startr = i;
                }
                if (j < startc) {
                    startc = j;
                }
                if (j > endc) {
                    endc = j;
                }
                endr = i;
            }
        }
    }

    // If white image
    if (endr < startr || endc < startc) {
        endr = small.shape[1];
        startr = 0;
        startc = 0;
        endc = small.shape[0];
    }

    var width = endc - startc + 1;
    var height = endr - startr + 1;
    var aspectRatio = width/height;

    var sizex = 0;
    var sizey = 0;

    if (width > height) {
        sizex = 20;
        sizey = 20 / aspectRatio;
    }
    else {
        sizey = 20;
        sizex = 20 * aspectRatio;
    }

    var box = tf.tensor([startr/img.shape[0], startc/img.shape[1], endr/img.shape[0], endc/img.shape[1]], [1, 4]);
    var smallTensor = tf.image.cropAndResize(img.expandDims(0).expandDims(-1), box, [0], [Math.floor(sizey), Math.floor(sizex)], 'bilinear');

    return tf.cast(smallTensor.squeeze(), 'int32');
}

// Inverts and normalizes the img values
function invert(img) {
    const one = tf.scalar(1);
    const norm = tf.scalar(255);

    return tf.sub(one, tf.div(img, norm));
}

function getCenter(img) {
    var sum = img.sum();

    var cx = tf.div(tf.sum(tf.matMul(img, tf.range(0, img.shape[1]).expandDims(-1))), sum);
    var cy = tf.div(tf.sum(tf.matMul(tf.range(0, img.shape[0]).expandDims(0), img)), sum);

    return {x: cx.bufferSync().get(0), y: cy.bufferSync().get(0)};
}