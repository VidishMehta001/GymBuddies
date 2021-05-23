import cv2
import math

class CheckRules:
    def __init__(self):
        self.previous_counter = 0
        self.count = 0
        self.voidFlag = False
        pass
    def check_squat_rules(self, pose_image, x, y, z, x_img, y_img, key_frame_detected, counter, brokenRules, writeNotes, decay, decaymsg):
        x_img = x_img.astype(int)
        y_img = y_img.astype(int)
        x0, y0 = x[0], y[0]
        x2, y2 = x[2], y[2]
        x4, y4 = x[4], y[4]
        x6, y6 = x[6], y[6]
        x8, y8 = x[8], y[8]
        x10, y10 = x[10], y[10]

        x11, y11 = x_img[0], y_img[0]
        x13, y13 = x_img[2], y_img[2]
        x15, y15 = x_img[4], y_img[4]
        x23, y23 = x_img[6], y_img[6]
        x25, y25 = x_img[8], y_img[8]
        x27, y27 = x_img[10], y_img[10]
        x28, y28 = x_img[11], y_img[11]

        img = pose_image
        countVoid = False
        angleKneeText = ""
        angleArmText = ""

        # Rules 1 (Angle of arm)
        angleArm = math.degrees(math.atan2(y2 - y0, x2 - x0) - math.atan2((y0-2) - y0, x0 - x0))
        if angleArm < 0:
            angleArm += 360
        angleArm = 360 - angleArm

        if angleArm < 125 and angleArm > 55:
            cv2.line(img, (x11, y11), (x13, y13), (0, 255, 0), 3)
            cv2.putText(img, str(int(angleArm)), (x11 - 50, y11 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        elif angleArm <55:
            cv2.line(img, (x11, y11), (x13, y13), (0, 0, 255), 3)
            cv2.putText(img, str(int(angleArm)), (x11 - 50, y11 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            angleArmText = "Arm too low. "
        else:
            cv2.line(img, (x11, y11), (x13, y13), (0, 0, 255), 3)
            cv2.putText(img, str(int(angleArm)), (x11 - 50, y11 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            angleArmText = "Arm too high. "


        # Rules 2 (Squat height)
        angleKnee = math.degrees(math.atan2(y10 - y8, x10 - x8) - math.atan2(y6 - y8, x6 - x8))
        if angleKnee < 0:
            angleKnee = 0 - angleKnee
        if key_frame_detected == 2:
            if angleKnee >90:
                cv2.line(img, (x23, y23), (x25, y25), (0, 0, 255), 3)
                cv2.line(img, (x25, y25), (x27, y27), (0, 0, 255), 3)
                cv2.putText(img, str(int(angleKnee)), (x25 - 50, y25 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                angleKneeText = "Bend Lower !! "
            else:
                cv2.line(img, (x23, y23), (x25, y25), (0, 255, 0), 3)
                cv2.line(img, (x25, y25), (x27, y27), (0, 255, 0), 3)
                cv2.putText(img, str(int(angleKnee)), (x25 - 50, y25 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        # # Rules 3 (Distance between Legs) - Not used. Performance too unstable.
        # cv2.line(img, (x27, y27), (x28, y28), (0, 0, 255), 3)
        # zo = (z11-z10)*1000
        #
        # # cv2.putText(img, str(int(z10)), (x27 - 50, y27 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        # # cv2.putText(img, str(int(z11)), (x28 - 50, y28 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        # cv2.putText(img, str(zo), (x28 - 50, y28 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)


        #Post-rules
        squatText = angleKneeText + angleArmText

        if len(squatText) != 0:

            if writeNotes == True:
                cv2.putText(img, squatText, (500, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (1000, 0, 255), 2)
            decaymsg = squatText
            decay = 24
            squatText = 'Rep ' + str(counter) + ': ' + squatText
            brokenRules.append(squatText)
            countVoid = True

        else:
            if writeNotes == True:
                if decay != 0:
                    cv2.putText(img, decaymsg, (400, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                (1000, 0, 255), 2)
                else:
                    cv2.putText(img, "Correct Posture. Keep Going!", (400, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (1000, 0, 255), 2)
            decay -= 1

        pose_image = self.counter_manage(counter, countVoid, img, writeNotes)

        return (pose_image, countVoid, brokenRules, decay, decaymsg, self.count)



    def check_pushup_rules(self, pose_image, x, y, z, x_img, y_img, key_frame_detected, counter, brokenRules, writeNotes, decay, decaymsg):
        x_img = x_img.astype(int)
        y_img = y_img.astype(int)
        x0, y0 = x[0], y[0]
        x2, y2 = x[2], y[2]
        x4, y4 = x[4], y[4]
        x6, y6 = x[6], y[6]
        x8, y8 = x[8], y[8]
        x10, y10 = x[10], y[10]

        x11, y11 = x_img[0], y_img[0]
        x13, y13 = x_img[2], y_img[2]
        x15, y15 = x_img[4], y_img[4]
        x23, y23 = x_img[6], y_img[6]
        x25, y25 = x_img[8], y_img[8]
        x27, y27 = x_img[10], y_img[10]

        img = pose_image
        countVoid = False
        angleArmPText = ""
        angleLegText = ""

        # Rules 1 (Angle of arm)
        angleArmP = math.degrees(math.atan2(y4 - y2, x4 - x2) - math.atan2(y0 - y2, x0 - x2))
        if angleArmP < 0:
            angleArmP += 360

        if key_frame_detected == 2:
            if angleArmP < 110:
                cv2.line(img, (x11, y11), (x13, y13), (0, 255, 0), 3)
                cv2.line(img, (x13, y13), (x15, y15), (0, 255, 0), 3)
                cv2.putText(img, str(int(angleArmP)), (x11 - 50, y11 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            else:
                cv2.line(img, (x11, y11), (x13, y13), (0, 0, 255), 3)
                cv2.line(img, (x13, y13), (x15, y15), (0, 0, 255), 3)
                cv2.putText(img, str(int(angleArmP)), (x11 - 50, y11 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                angleArmPText = "Bend Lower !!  "

        if key_frame_detected == 1:
            if angleArmP > 130:
                cv2.line(img, (x11, y11), (x13, y13), (0, 0, 255), 3)
                cv2.line(img, (x13, y13), (x15, y15), (0, 0, 255), 3)
                cv2.putText(img, str(int(angleArmP)), (x11 - 50, y11 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0),
                            2)
            else:
                cv2.line(img, (x11, y11), (x13, y13), (0, 255, 0), 3)
                cv2.line(img, (x13, y13), (x15, y15), (0, 255, 0), 3)
                cv2.putText(img, str(int(angleArmP)), (x11 - 50, y11 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255),
                            2)
                angleArmPText = "Raise your body higher !! "

        # Rules 2 (Leg Straight)
        angleLeg = math.degrees(math.atan2(y10 - y8, x10 - x8) - math.atan2(y6 - y8, x6 - x8))
        if angleLeg < 0:
            angleLeg += 360

        if angleLeg > 145 and angleLeg < 215:
            cv2.line(img, (x23, y23), (x25, y25), (0, 255, 0), 3)
            cv2.line(img, (x25, y25), (x27, y27), (0, 255, 0), 3)
            cv2.putText(img, str(int(angleLeg)), (x25 - 50, y25 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        else:
            cv2.line(img, (x23, y23), (x25, y25), (0, 0, 255), 3)
            cv2.line(img, (x25, y25), (x27, y27), (0, 0, 255), 3)
            cv2.putText(img, str(int(angleLeg)), (x25 - 50, y25 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            angleLegText = "Straighten your leg !! "

        #Post-Rules

        pushUpText = angleArmPText + angleLegText

        if len(pushUpText) != 0:
            if writeNotes == True:
                cv2.putText(img, pushUpText, (100, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (1000, 0, 255), 2)
            decaymsg = pushUpText
            decay = 24
            pushUpText = 'Rep ' + str(counter) + ': ' + pushUpText
            brokenRules.append(pushUpText)
            countVoid = True
        else:
            if writeNotes == True:
                if decay != 0:
                    cv2.putText(img, decaymsg, (400, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                (1000, 0, 255), 2)
                else:
                    cv2.putText(img, "Correct Posture. Keep Going!", (400, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (1000, 0, 255), 2)
            decay -= 1

        pose_image = self.counter_manage(counter, countVoid, img, writeNotes)


        return (pose_image, countVoid, brokenRules, decay, decaymsg, self.count)

    def counter_manage(self, counter, countVoid, img, writeNotes):

        if countVoid == True:
            self.voidFlag = True

        if counter > self.previous_counter:
            self.previous_counter = counter
            if self.voidFlag == False:
                self.count += 1
            else:
                self.voidFlag = False

        Text = "Valid Count:" + str(self.count)
        Text2 = "Total Count:" + str(counter)
        if writeNotes == True:
            cv2.putText(img, Text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (1000, 0, 255), 2)
            cv2.putText(img, Text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (1000, 0, 255), 2)
        return img






