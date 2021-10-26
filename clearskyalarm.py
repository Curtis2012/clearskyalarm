#
#  clear-sky-alarm
#
#  2021-09-20 C. Collins created
#  2021-09-26 C. Collins implement watch allsky cam image directory for new files
#  2021-10-25 C. Collins moved to github
#
#  todo:
#
#         - FIX: syntax of all sys.stderr.write() calls, test
#         - Add detected file house keeping, delete files after x days...
#         - Add multiple notification channels: push via BT, buzzer,... each enabled/disabled by config flag
#         - Add push notifications via service
#
#

import os
import sys
import time
import math
import numpy as np
import numpy.core.multiarray
from numpy import ndarray
import cv2
import json
import pyinotify
import requests

imageFile = ""


class ConfigClass:

    def buildNotifyPayload(self):

        # smsPayload format:
        #
        # resp = requests.post('https://textbelt.com/text', {
        #     'phone': '5555555555',
        #     'message': 'Hello world',
        #     'key': 'textbelt',
        # })

        config.smsPayload = {'phone': config.smsPhone, 'message': config.smsMsg, 'key': config.smsAPIKey}

        if (config.debug):
            print("config.smsPayload = ", config.smsPayload)
        return

    def loadConfig(self):
        try:
            with open('/home/pi/clearskyalarm/clearskyalarmconfig.json', 'r') as f:
                self.configFile = json.load(f)
                print(json.dumps(self.configFile, indent=4, sort_keys=True))
                self.debug = self.configFile['debug']
                self.starCountThreshold = self.configFile['starCountThreshold']
                self.distanceThreshold = self.configFile['distanceThreshold']
                self.detectionThreshold = self.configFile['detectionThreshold']
                self.templatePath = self.configFile['templatePath']
                self.imagePath = self.configFile['imagePath']
                self.detectedPath = self.configFile['detectedPath']
                self.detectedTag = self.configFile['detectedTag']
                self.writeDetectedFile = self.configFile['writeDetectedFile']
                self.smsAPIKey = self.configFile['smsAPIKey']
                self.smsPhone = self.configFile['smsPhone']
                self.smsRegion = self.configFile['smsRegion']
                self.smsMsg = self.configFile['smsMsg']
                self.notifySMS = self.configFile['notifySMS']
                self.notifyDelta = self.configFile['notifyDelta']
                self.imageTag = self.configFile['imageTag']
                self.imageType = self.configFile['imageType']

                self.notifyTimeStamp = time.time()
                self.buildNotifyPayload()
        except:
            sys.stderr.flush()
            sys.exit("Error opening or parsing config file, exiting\n")

        try:
            self.template = cv2.imread(self.templatePath, cv2.IMREAD_GRAYSCALE)
            self.w, self.h = config.template.shape[::-1]
        except:
            sys.stderr.write("Failed to load template file\n")
            return (False)

        print("Clear-sky-alarm, configuration file loaded...")
        return (True)


config = ConfigClass()


def allskyFile():
    result = False

    if ((imageFile.find(config.imageTag) != -1) and (imageFile.find(config.imageType) != -1)): result = True  # check for allskycam image file
    if (imageFile.find(config.detectedTag) != -1): result = False   # verify not a clear sky alarm "detected" file
    if (imageFile.find("thumbnails") != -1): result = False         # verify not a thumbnail

    return (result)


class HandleNotifyClass(pyinotify.ProcessEvent):
    def process_IN_CREATE(self, event):
        global imageFile

        imageFile = os.path.join(event.path, event.name)
        print("Notification of new file: %s " % imageFile)
        config.detectedFile = config.detectedTag + event.name

        if (allskyFile()):
            try:
                time.sleep(5)  # kludge: delay after create until I come up with a better way to detect file ready
                if (not countStars()):
                    sys.stderr.write("Star count failed\n")
            except:
                sys.stderr.write("Error performing star count\n")
        else:
            print("File is not an allskycam image file, ignored.")

        return (True)


def notifyDeltaExpired():
    elapsed = time.time() - config.notifyTimeStamp
    if (config.debug):
        print("config.notifyDelta = ", config.notifyDelta)
        print("elapsed since last notification = ", elapsed)

    if (elapsed > config.notifyDelta):
        config.notifyTimeStamp = time.time()
        return (True)

    return (False)


def smsNotifyError(response):
    sys.stderr.write("\nSMS notification send failed\n")
    sys.stderr.write("\nconfig.smsPayload = %s" % config.smsPayload)
    sys.stderr.write("\nresponse = %s" % response.json())


def notify():
    if (not notifyDeltaExpired()):
        return

    if (config.notifySMS):
        try:
            if (config.debug): print("config.smsPayload) = ", config.smsPayload)
            resp = requests.post('https://textbelt.com/text', config.smsPayload)
            if (config.debug): print(resp.json())
            resp_dict = resp.json()
            if (resp_dict['success']):
                print("\nSMS notification sent")
                print("\nSMS quota remaining = ", resp_dict['quotaRemaining'])
                if (resp_dict['quotaRemaining'] < 1):
                    print("\nSMS quota exceeded, top up!")
            else:
                smsNotifyError(resp)
        except:
            smsNotifyError(resp)

    else:
        print("SMS notification set to false in config file, no notification sent")


def countStars():
    global config

    print("Counting stars...")
    startTime = time.time()

    try:
        img = cv2.imread(imageFile)
    except:
        sys.stderr.write("Failed to load star image file\n")
        sys.stderr.write("Image file path = %s" % imageFile)
        return (False)

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    try:
        result = cv2.matchTemplate(gray_image, config.template, cv2.TM_CCOEFF_NORMED)
    except:
        sys.stderr.write("Star template match failed\n")
        return (False)

    loc = np.where(result >= config.detectionThreshold)

    numStars = 0
    ptPrev = loc[0]  # fix to reference just first tuple
    for pt in zip(*loc[::-1]):
        d = math.sqrt(((pt[0] - ptPrev[0]) ** 2) + ((pt[1] - ptPrev[1]) ** 2))
        if (d > config.distanceThreshold):
            cv2.rectangle(img, pt, (pt[0] + config.w, pt[1] + config.h), (0, 255, 255), 1)
            numStars += 1
        ptPrev = pt

    try:
        if (config.writeDetectedFile):
            os.chdir(config.detectedPath)
            if (cv2.imwrite(config.detectedFile, img, params=None)):
                if (config.debug):
                    print("Detected file written: ", config.detectedFile)
            else:
                sys.stderr.write(
                    "Error writing detected image\n")  # non-critical error so do not return a fail on deteced file write
    except:
        sys.stderr.write("Error accessing detected file/file path\n")  # non-critical error so do not return a fail on deteced file write

    print("star count =", numStars)
    elapsedTime = time.time() - startTime
    print("Star count elapsed time = ", elapsedTime, " seconds")

    if (numStars > config.starCountThreshold):
        try:
            notify()
        except:
            sys.stderr.write("Error sending notification\n")

    return (True)


def main():
    config.loadConfig()

    print("Creating watch for new files on ", config.imagePath)
    wm = pyinotify.WatchManager()
    mask = pyinotify.IN_CREATE
    notifier = pyinotify.Notifier(wm, HandleNotifyClass())
    wm.add_watch(config.imagePath, mask, rec=True)

    print("Watching for new file events...")
    notifier.loop()


if __name__ == "__main__":
    main()
