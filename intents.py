# This file defines the final intent categories for AppleSupport, decided
# after reading through clustered samples of real customer messages.
# See discover_intents.py for how these were derived.
#
# Revised after checking sample classifications: split the old
# "messaging_keyboard_bug" (which the model was using as a catch-all for
# anything texting-related) into a tighter "keyboard_input_bug" category,
# and added "file_transfer_issue" as its own category since it was getting
# wrongly absorbed into sync_icloud_issue.

INTENTS = {

    "battery_charging_issue": {
        "description": "Device won't charge, charges slowly, or battery drains unusually fast",
        "examples": [
            "So the phone is now dead and won't charge, regardless of what charging cable I use or what I plug it into. What are you going to do about that?",
            "Yes, I've done all that and tried several Apple approved lightning cords. It charges very slowly or not at all.",
            "The iPod Touch 6th generation is now having the worst battery life."
        ]
    },

    "connectivity_issue": {
        "description": "WiFi, Bluetooth, GPS, or cellular network not connecting or dropping out",
        "examples": [
            "No - consistently it won't even recognise any Wi-fi - I've reported this problem multiple times and nothing has changed.",
            "Maps is also having GPS problems, it will work for a minute then drop out of location.",
            "notification alerts are not mirrored in watch(v4). Working fine till Oct 08."
        ]
    },

    "app_performance_issue": {
        "description": "Apps crashing, freezing, or the device running slowly, especially after an update",
        "examples": [
            "Why does it take 10-15 seconds to open any app? The camera app even?",
            "iPhone crashing constantly since last night.",
            "your iOS 11 is buggy, sticky, and slow."
        ]
    },

    "sync_icloud_issue": {
        "description": "Photos, contacts, or files failing to sync or back up through iCloud specifically (not local file transfer via cable or iTunes)",
        "examples": [
            "It's just that I can't download anything. There's no button in my iCloud settings that allow the downloading process to begin.",
            "When I open Photos, Albums, Camera Roll, the most recent photos shown is dated 10/1/17. The next most recent is from 2014.",
            "I've tried that before. They aren't coming up on my Find My iPhone app."
        ]
    },

    "file_transfer_issue": {
        "description": "Problems moving files between the device and a computer or car, via USB, iTunes, or cable — not cloud sync",
        "examples": [
            "I updated my itunes and now I can't upload pdfs to my ipad from my PC",
            "Forgot how insanely frustrating iphone is when it comes to transferring files from my phone.",
            "why can't I play music from my phone in my car via USB anymore"
        ]
    },

    "audio_media_issue": {
        "description": "Sound, speaker, screenshot, or media playback problems (not file transfer)",
        "examples": [
            "My speaker phone hasn't worked since the software update.",
            "I have upgraded to iOS 11.1.2, now taking screenshot and sharing takes a lot of time.",
            "It's not displaying anything when I listen to music or a podcast."
        ]
    },

    "billing_issue": {
        "description": "Unwanted charges, subscription confusion, or refund requests",
        "examples": [
            "Was dl on accident. Do I report it as a problem? I just want the $1 or so back to dl the right song without having one I don't want.",
            "Looked at my account, for some reason I'm paying £9.99 for music I never agreed to or set up.",
            "My gripe is not that I don't know how to return items, it's that the colour advertised is very different from the finished product."
        ]
    },

    "keyboard_input_bug": {
        "description": "Bugs specifically in typing, autocorrect, or the keyboard itself — not spam texts or messaging app crashes",
        "examples": [
            "Wassup with the letter i on our keyboards yo",
            "pull it together. iOS 11 is a text-glitch nightmare.",
            "Come on this letter I thing is ridiculous."
        ]
    },

    "general_bug_other": {
        "description": "Miscellaneous bugs, store/policy questions, or anything that doesn't clearly fit the categories above",
        "examples": [
            "I think I've found the fix, but there's a bug in the Clock app or part of the OS.",
            "Are all your physical stores oriented to working with reverse logistics?",
            "I keep getting spam texts, I've blocked the contact 4 times but it doesn't stop."
        ]
    },
}