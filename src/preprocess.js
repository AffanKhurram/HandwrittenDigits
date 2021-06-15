
function process(canv) {
    var tensor = tf.browser.fromPixels(canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height)).mean(2);
    var small = resize(tensor);
    var normalized = invert(small);
    var center = getCenter(normalized);

    var before_x = 14 - center.x;
    var before_y = 14 - center.y;
    var after_x = 28 - (before_x + normalized.shape[1]);
    var after_y = 28 - (before_y + normalized.shape[0]);

    var finalImage = normalized.pad([[before_y, after_y], [before_x, after_x]]);
    return finalImage;
}

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