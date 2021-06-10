
var canvas = document.getElementById('canvas');
var sigpad = new SignaturePad(canvas, {
    backgroundColor: 'rgba(255, 255, 255, 0)'
});
var clearButton = document.getElementById('clear');
var saveButton = document.getElementById('save');

saveButton.addEventListener('click', function(event) {
    var data = sigpad.toDataURL('image/png');
    console.log(data);
});

clearButton.addEventListener('click', function(event) {
    sigpad.clear();
});


model = tf.loadLayersModel('tfjs_model/model.json');
model.then(function (res) {
    tensor = tf.zeros([1, 28, 28, 1]);
    result = res.predict(tensor)
    console.log(result.bufferSync().get(0, 1))
});