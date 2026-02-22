import pyttsx3
def speak(text):
    engine = pyttsx3.init()  # 初始化语音引擎
    engine.say(text)         # 将文本添加到语音队列中
    engine.runAndWait()      # 运行语音队列中的命令，并等待它们完成
#     print("engine id:", id(engine))
#
# # 使用函数
# for i in range(3):
#
#     speak("hello world")

for i in range(10):
    print(i)

    text = "hello world"
    engine = pyttsx3.init()  # 初始化语音引擎
    engine.say(text)         # 将文本添加到语音队列中
    engine.runAndWait()      # 运行语音队列中的命令，并等待它们完成
    engine.stop()
    print("engine id:", id(engine))
    del engine
