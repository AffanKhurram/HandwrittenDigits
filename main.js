'use strict';

var canvas = document.getElementById('canvas');
var sigpad = new SignaturePad(canvas, {
    backgroundColor: 'rgb(255, 255, 255)',
    minWidth: 4,
    maxWidth: 4
});

var save = document.getElementById('save');
var clear = document.getElementById('clear');

save.addEventListener('click', function(event) {
    console.log('original img', sigpad.toDataURL());
    var trimCanv = document.createElement('canvas');
    trimCanv.width = canvas.width;
    trimCanv.height = canvas.height;
    trimCanv.getContext('2d').drawImage(canvas, 0, 0);
    trim(trimCanv);

    var smallImg = new Image();
    smallImg.src = trimCanv.toDataURL();

    smallImg.onload = function() {
        var aspectRatio = smallImg.width/smallImg.height;
        if (smallImg.width > smallImg.height) {
            trimCanv.width = 20;
            trimCanv.height = trimCanv.width / aspectRatio;
        }
        else {
            trimCanv.height = 20;
            trimCanv.width = trimCanv.height * aspectRatio;
        }
        var smallCtx = trimCanv.getContext('2d');
        smallCtx.fillStyle = 'white';
        smallCtx.fillRect(0, 0, trimCanv.width, trimCanv.height);
        smallCtx.drawImage(smallImg, 0, 0, trimCanv.width, trimCanv.height);
        console.log(trimCanv.toDataURL());
        console.log(trimCanv.width, trimCanv.height);

        var imgData = smallCtx.getImageData(0, 0, trimCanv.width, trimCanv.height);
        console.log('creation:', imgData);
        grayscale(imgData);
        console.log('after grayscale: ', imgData);
        var c = center(imgData);

        var finalCanv = document.createElement('canvas');
        var finalCtx = finalCanv.getContext('2d');
        finalCanv.width = 28;
        finalCanv.height = 28;
        var dx = 14 - c.x;
        var dy = 14 - c.y;
        finalCtx.fillStyle = 'white';
        finalCtx.fillRect(0, 0, 28, 28);
        finalCtx.putImageData(imgData, dx, dy);
        console.log(finalCanv.toDataURL());

        var model = tf.loadLayersModel('tfjs_model/model.json');
        model.then(function (res) {
            var tensor = tf.browser.fromPixels(finalCtx.getImageData(0, 0, 28, 28)).mean(2).toFloat().expandDims(0).expandDims(-1);
            var norm = tf.scalar(255);
            var sub = tf.scalar(1);
            tensor = tf.div(tensor, norm);
            tensor = tf.sub(sub, tensor);

            var tensorImage = document.createElement('canvas');
            tf.browser.toPixels(tensor.reshape([28, 28, 1]), tensorImage).then(function (res) {
                console.log('tensor', tensorImage.toDataURL());
            });

            var result = res.predict(tensor.reshape([1, 28, 28, 1]));
            result.print();
            result.argMax(1).print();
        });
    }

    // img.onload = function() {
    //     var small = document.createElement('canvas');
    //     var size = 20;  
    //     small.width = size;
    //     small.height = size;
    
    //     var ctx = small.getContext('2d');
    //     ctx.fillStyle = 'white';
    //     ctx.fillRect(0, 0, size, size);
    //     ctx.drawImage(img, 0, 0, size, size);
    //     console.log(small.toDataURL());
    // }
    
});

clear.addEventListener('click', function(event) {
    sigpad.clear()
});

function trim(canv) {
    var img = canv.getContext('2d').getImageData(0, 0, canv.width, canv.height);
    var data = img.data;
    var startx = img.width;
    var endx = 0;
    var starty = img.height;
    var endy = 0;

    for (var i = 0; i < img.height; i++) {
        for (var j = 0; j < img.width; j++) {
            var valIndex = (i * img.width + j)*4;
            if (data[valIndex] !== 255 && data[valIndex+1] !== 255 && data[valIndex+2] !== 255) {
                if (i < starty) {
                    starty = i;
                }
                if (j < startx) {
                    startx = j;
                }
                if (j > endx) {
                    endx = j;
                }
                endy = i;
            }
        }
    }

    var smallImg = canv.getContext('2d').getImageData(startx, starty, endx-startx + 1, endy-starty + 1);
    canv.width = smallImg.width;
    canv.height = smallImg.height;
    canv.getContext('2d').putImageData(smallImg, 0, 0);
}

function grayscale(imgData) {
    var width = imgData.width;
    var height = imgData.height;
    var data = imgData.data;
    for (var i = 0; i < height; i++) {
        for (var j = 0; j < width; j++) {
            var valIndex = 4 * (i * width + j);
            var avg = (data[valIndex] + data[valIndex+1] + data[valIndex+2])/3
            data[valIndex] = avg;
            data[valIndex+1] = avg;
            data[valIndex+2] = avg;
        }
    }
}

function center(imgData) {
    var width = imgData.width;
    var height = imgData.height;
    var data = imgData.data;
    var x = 0;
    var y = 0;
    var sum = 0;
    for (var i = 0; i < height; i++) {
        for (var j = 0; j < width; j++) {
            var valIndex = 4 * (i * width + j);
            var val = 1 - data[valIndex]/255;
            sum += val;
            x += j * val;
            y += i * val;
        }
    }
    x /= sum;
    y /= sum;

    return {x: x, y: y};
}