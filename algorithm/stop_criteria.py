
class StopCriteria(object):
    """
    The object to compute and determine whether the growing should stop
    Attributes
    metric: str
        A description for the metric type
    stop: boolean
        Indicate the growing status: False or True
    Methods
    compute(self, region, image,threshold)
        determine whether the growing should stop

    """

    def __init__(self, criteria_metric='size', stop=False):
        """
        Parameters
        criteria_metric: str, optional
            A description for the metric type. The supported types include 'homogeneity','size','gradient'.
            Default is 'size'
        """
        if criteria_metric == 'size' or criteria_metric == 'homogeneity' or criteria_metric == 'gradient':
            self.metric = criteria_metric

        self.stop = stop

    def set_metric(self, metric):
        """
        Set the name of the stop criteria..
        """
        self.metric = metric

    def get_metric(self):
        """
        Get the name of the stop criteria..
        """
        return self.metric

    def compute(self, region, image, threshold=None):
        """
        compute the metric of region according to the region and judge whether the metric meets the stop threshold
        Parameters
        image: numpy 2d/3d/4d array
            The numpy array to represent 2d/3d/4d image to be segmented.
        region: Region object
            It represents the current region and associated attributes
        threshold: float, optional
            The default is None which means the adaptive method will be used.
        """

        if self.metric == 'size':
            if region.label_size() >= threshold:
                self.stop = True

    def isstop(self):
        return self.stop

    def set_stop(self):
        """
        Reset the stop signal
        """
        self.stop = False

class MultiSeedsStopCriteria(StopCriteria):
    """
    The object to compute and determine whether the growing should stop
    Attributes
    metric: str
        A description for the metric type
    stop: boolean
        Indicate the growing status: False or True
    Methods
    compute(self, region, image,threshold)
        determine whether the growing should stop

    """

    def __init__(self, criteria_metric='size', stop=False):
        """
        Parameters
        criteria_metric: str, optional
            A description for the metric type. The supported types include 'homogeneity','size','gradient'.
            Default is 'size'
        """
        if criteria_metric == 'size' or criteria_metric == 'homogeneity' or criteria_metric == 'gradient':
            self.metric = criteria_metric

        self.stop = stop

    def compute(self, regions, image, threshold=None):
        """
        compute the metric of region according to the region and judge whether the metric meets the stop threshold
        Parameters
        image: numpy 2d/3d/4d array
            The numpy array to represent 2d/3d/4d image to be segmented.
        region: Region object
            It represents the current region and associated attributes
        threshold: float, optional
            The default is None which means the adaptive method will be used.
        """
        neighbors_sum = 0
        labels_sum = 0
        for region in regions:
            neighbors_sum += region.get_neighbor().shape[0] #The neighbors has been labeled.
            labels_sum += region.get_label().shape[0]

        if (threshold and labels_sum < threshold) or neighbors_sum == 0:
            self.stop = True

