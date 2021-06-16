'use strict';

async function displayConv(model) {
    var conv = model.getLayer("", 1);
    var filters = conv.getWeights()[0].squeeze();
    var first = filters.slice([0,0,0], [3, 3, 1]);
    var norm = normalize(first);
    
    var displayCanv = document.createElement('canvas');
    var displayCtx = displayCanv.getContext('2d');
    displayCanv.width = 350;
    displayCanv.height = 350;
    var smallCanvs = document.createElement('canvas');
    var smallCtx = smallCanvs.getContext('2d');
    for (var i = 0; i < 8; i++) {
        var filter = filters.slice([0, 0, i], [3, 3, 1]);
        var norm = normalize(filter);
        var big = tf.image.resizeNearestNeighbor(norm, [99, 99]);


        await tf.browser.toPixels(big, smallCanvs);
        var x = (i % 3) * 109;
        var y = Math.floor(i/3) * 109;
        console.log('filter', i, ':', smallCanvs.toDataURL());
        displayCtx.putImageData(smallCtx.getImageData(0, 0, 99, 99), x, y);
        
    }
    console.log(displayCanv.toDataURL());
        
}

function normalize(filter) {
    var max = filter.max();
    var min = filter.min();
    var norm = tf.div(tf.sub(filter, min), tf.sub(max, min));

    return norm;
}