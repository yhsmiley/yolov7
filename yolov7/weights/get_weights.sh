#!/bin/bash
for arch in $@
do
    case $arch in
        yolov7-tiny)
            wget -O yolov7-tiny_state.pt https://xsoftware15-my.sharepoint.com/:u:/g/personal/smiley_c8r6_onmicrosoft_com/IQCKcfD9VojLSb_t5cVyla0eAWrvZGojzbdYVXVXrLmtX3Q?e=NzBpdy;;
        yolov7)
            wget -O yolov7_state.pt https://xsoftware15-my.sharepoint.com/:u:/g/personal/smiley_c8r6_onmicrosoft_com/IQDZe2zMOCn5SbjTvKzRlEzJAbwWRvxK1A1f4SqBZVn3K4c?e=dWq7xe;;
        yolov7x)
            wget -O yolov7x_state.pt https://xsoftware15-my.sharepoint.com/:u:/g/personal/smiley_c8r6_onmicrosoft_com/IQC8vc9WmMIoTovcco0NczuIAdecgNk4yNU0PenH0UBjfuo?e=bYT0DH;;
        yolov7-w6)
            wget -O yolov7-w6_state.pt https://xsoftware15-my.sharepoint.com/:u:/g/personal/smiley_c8r6_onmicrosoft_com/IQDrk27m938xTYGp-j_lcqj6AcN7G4l2rvaTvyP6aZ4CHk8?e=5W8wKK;;
        yolov7-e6)
            wget -O yolov7-e6_state.pt https://xsoftware15-my.sharepoint.com/:u:/g/personal/smiley_c8r6_onmicrosoft_com/IQB1kTsJ78TsRIVI4egroHVtARgEKNykuBra6uvH0B4Gmx0?e=qszD1l;;
        yolov7-d6)
            wget -O yolov7-d6_state.pt https://xsoftware15-my.sharepoint.com/:u:/g/personal/smiley_c8r6_onmicrosoft_com/IQDmCZlqavMITokKEdzdeg6MAQJ9qQuaXmK4nAO4XlUG3Fg?e=XtFUp7;;
        yolov7-e6e)
            wget -O yolov7-e6e_state.pt https://xsoftware15-my.sharepoint.com/:u:/g/personal/smiley_c8r6_onmicrosoft_com/IQBZC2Y-AxhNS74elyMQbTUcAXjX63F4nBlh63xuCHegtzQ?e=ntPtdl;;
        *)
            echo $arch weights are not available;;
    esac
done
