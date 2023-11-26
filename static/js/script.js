function sendContentToApp(default_img_url){
    var img = document.getElementById("img_url").value
    var xhr = new XMLHttpRequest();
    var url = window.location.href + '/result';

    console.log(url)

    // Set up the request
    xhr.open('POST', url, true);
    xhr.setRequestHeader('Content-Type', 'application/json');

    // Set up the callback function for when the request completes
    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4 && xhr.status === 200) {
            // Handle the response from the Flask server if needed
            // console.log('Response:', xhr.responseText);
            document.getElementById('result').innerHTML = xhr.responseText;
        }
    };

    if (img == ""){
        img = default_img_url
        document.getElementById("img_url").value = img
    }

    // Prepare the data to be sent as JSON
    var jsonData = JSON.stringify(img);
    console.log(jsonData);

    // Send the request with the captured data
    xhr.send(jsonData);
}

function clearContent() {
    document.getElementById("result").innerHTML = ""
    document.getElementById("img_url").value = ""
}