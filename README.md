# inkquick

## Introduction
This is my capstone project for the data science immersive program at [Galvanize](http://www.galvanize.com/courses/data-science/#.VdzRf1NViko).
The idea was to predict a tattoo artist for a user submitted image of a tattoo. I achieved this using a convolutional neural network
to identify images of tattoos from artists instagram accounts. I then used the penultimate layer of that network as a feature vector for
comparing images. Using cosine similarity I could compare the collected tattoos of an artist to the user submitted image, and then
suggest artists with highest average similarity.

## Outline

* Use Yelp and Instagram APIs to collect a group of tattoo artists Instagram accounts in the San Francisco bay area.
* Code images from the accounts as whether or not they are pictures of tattoos. (Used Flask app to allow friends to help code images)
* Train neural network on coded images to create model for filtering images. (Achieved ~85% accuracy)
* Filter from ~500 Instagram accounts and ~500,000 photos to ~180 accounts and ~65,000 photos. (Account filtering removed non-tattoo artists and artists not from bay area)
* Run submitted images through neural network to extract feature vector for comparison to artists photos.
* Suggest top 10 artists with highest average cosine similarity between their work and submitted photo.


