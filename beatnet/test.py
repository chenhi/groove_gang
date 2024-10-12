import simpleaudio, time, vlc
strong_beat = simpleaudio.WaveObject.from_wave_file('strong_beat.wav')
#while True:
strong_beat.play()
p = vlc.MediaPlayer('inputs/lms.mp3')
p.play()
#time.sleep(0.5)
#strong_beat.play()
#time.sleep(0.5)
#strong_beat.play()
#time.sleep(0.5)

