var express = require('express');
var router = express.Router();
var axios = require('axios')

var app = require('express')();
var http = require('http').Server(app);
var io = require('socket.io')(http);
app.use(express.static(__dirname + '/../static'));

console.log(__dirname)
console.log(__dirname + '/../static')
/* GET home page. */
router.get('/', function(req, res, next) {
});

app.get('/', function(req, res){
  res.sendFile(__dirname + '/index.html');
});

io.on('connection',async function(socket){
  //getTweets();
  var dict = await getTweets("http://localhost:8983/solr/tweets/select?indent=on&q=ml_tag:0&25&sort=tweet_risk%20desc&rows=25&wt=json")
  //console.log("sending tweets: " + dict.length);
  socket.emit("tweetIn", JSON.stringify(dict));


  socket.on('updateTweets',async function(query){
    console.log(query);
    var dict = await getTweets(query);
    //console.log("updating tweets");
    socket.emit("tweetIn", JSON.stringify(dict));
  });

  socket.on('updateTags',async function(query){
    console.log(query);
    var dict = await getTags(query);
    socket.emit("tagsIn", JSON.stringify(dict));
  });

  socket.on('commonlyUsed',async function(input){
    var queries = JSON.parse(input);
    
    for(i = 0;i < queries.length; i++){
      var dict = await getUsedWith(queries[i]);
      console.log(dict);
      socket.emit("commonlyusedIn", JSON.stringify(dict));
    }
  });

});

function getTags(query){
  var dict = [];
  var tweets;
  return new Promise((resolve,reject) => {
    axios.get(query).then(function (response) {
      tags = response.data.response.docs;
      tags.forEach(function(tags){
      dict.push([tags.tag_text,tags.frequency]);
      }, this);
      //console.log(dict);
      resolve(dict);
    })
    .catch(function (error) {
      console.log(error);
      reject(error)
    });

  })
}

function getUsedWith(query){
  var dict = [];
  var tweets;
  return new Promise((resolve,reject) => {
    axios.get(query).then(function (response) {
      var common = response.data.response.docs;
      common.forEach(function(common){
      dict.push([common.tag_text]);
      }, this);
      resolve(dict);
    })
    .catch(function (error) {
      console.log(error);
      reject(error)
    });

  })
}

function getTweets(query){
  var dict = [];
  var tweets;
  return new Promise((resolve,reject) => {
    axios.get(query).then(function (response) {
      tweets = response.data.response.docs;
      tweets.forEach(function(tweet){
      dict.push([tweet.user_screen_name,tweet.status_text,tweet.created_at.replace("T"," ").replace("Z"," "),tweet.user_location_coordinates,tweet.id]);//Throws a tantrum with |
      }, this);
      resolve(dict);
    })
    .catch(function (error) {
      console.log(error);
      reject(error)
    });

  })
}

http.listen(3000, function(){
  console.log('listening on *:3000');
});
module.exports = router;