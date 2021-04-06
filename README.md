# background-subtraction

The purpose of background subtraction is to demonstrate competency in distinguishing the background and foreground in videos in real time. This code uses two built-in methods from OpenCV as well as a custom method which relies on contour sizes. This code accopmanies my article: https://towardsdatascience.com/background-removal-with-python-b61671d1508a

When creating a new Video object, use 0 to access the first available web camera. Otherwise use the path to a saved video. There are two main methods: removeBackgroundCV() and removeBackground().

Using removeBackgroundCV() relies on either of two OpenCV background removal methods: MOG2 and KNN. Either of these methods work similarly insofar that they remove backgrounds based on changes in the environment over time. These will work well in particular to isolate a moving foreground across a static background. 

Using removeBackground() uses a method that relies on contour size. Essentially, any contour considered too big or too small will be rendered the background and anything in between as the foreground. This will work well when trying to isolate the foreground that either aren't moving or don't move move much. Some fine-tuning in the parameters may have to be used to get a good result. For the best results, use a plain, non-busy background, such as a white wall or green screen.

Using either of these two methods will open a new window with the background removed.

As a demonstration of use:
example = Video(0)

# This uses the OpenCV method MOG2
example.removeBackgroundCV("MOG2")

# This uses the OpenCV method KNN
example.removeBackgroundCV("KNN")

# This uses the contour method
example.removeBackground(
    canny_low = 15, 
    canny_high = 150,
    min_area = 0.0005,
    max_area = 0.95
)
