# requirements
- 모델 다운 -> [모델](https://drive.google.com/file/d/161oyQcYpdkVdw8gKz_MA8RD-Wtg9XDp3/view)
- 다운 받은 모델 `denoising/models`에 넣기
- generator 다운 -> [generator](https://drive.google.com/file/d/1qmxUEKADkEM4iYLp1fpPLLKnfZ6tcF-t/view)
- 다운 받은 generator.zip `networks/`에 넣기
# start
```
    $ python3 -m venv venv
    $ source venv/bin/activate
    $ pip install -r requirements.txt
    $ python inference.py -p <이미지 폴더 주소>
```
