//! Windows: windows from EnumWindows, pictures with GDI (what the screen shows in the window's
//! drawing area, shrunk by StretchBlt), virtual input with SendInput (scan codes, relative
//! mouse movement), and the emergency stop keys through a low-level keyboard hook.
//!
//! Windows marks this input as injected; Tools does not hide that. A game running as
//! administrator does not receive input from a program that isn't (Windows' UIPI).
use super::{Desktop, Rect, WindowInfo};
use crate::live::controls::{Button, KeyInfo};
use crate::Res;
use std::ptr::null_mut;
use std::sync::mpsc;
use std::sync::OnceLock;
use windows_sys::Win32::Foundation::{CloseHandle, HWND, LPARAM, LRESULT, POINT, RECT, WPARAM};
use windows_sys::Win32::Graphics::Gdi::{
    ClientToScreen, CreateCompatibleDC, CreateDIBSection, DeleteDC, DeleteObject, GetDC, ReleaseDC,
    SelectObject, SetBrushOrgEx, SetStretchBltMode, StretchBlt, BITMAPINFO, BITMAPINFOHEADER, BI_RGB,
    DIB_RGB_COLORS, HALFTONE, SRCCOPY,
};
use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
use windows_sys::Win32::System::Threading::{
    OpenProcess, QueryFullProcessImageNameW, PROCESS_NAME_WIN32, PROCESS_QUERY_LIMITED_INFORMATION,
};
use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
    GetAsyncKeyState, SendInput, INPUT, INPUT_0, INPUT_KEYBOARD, INPUT_MOUSE, KEYBDINPUT,
    KEYEVENTF_EXTENDEDKEY, KEYEVENTF_KEYUP, KEYEVENTF_SCANCODE, MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP,
    MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP, MOUSEEVENTF_MOVE, MOUSEEVENTF_RIGHTDOWN,
    MOUSEEVENTF_RIGHTUP, MOUSEINPUT, VK_CONTROL, VK_END, VK_MENU,
};
use windows_sys::Win32::UI::WindowsAndMessaging::{
    CallNextHookEx, EnumWindows, GetClientRect, GetForegroundWindow, GetMessageW, GetWindow,
    GetWindowLongPtrW, GetWindowTextLengthW, GetWindowTextW, GetWindowThreadProcessId, IsIconic, IsWindow,
    IsWindowVisible, SetForegroundWindow, SetWindowsHookExW, ShowWindow, GWL_EXSTYLE, GW_OWNER, HC_ACTION,
    KBDLLHOOKSTRUCT, MSG, SW_RESTORE, WH_KEYBOARD_LL, WM_KEYDOWN, WM_SYSKEYDOWN, WS_EX_TOOLWINDOW,
};

/// Marks Tools' own input (`dwExtraInfo`), so Tools can tell it from a person's later.
pub const MARK: usize = 0x5945_4E44;

pub struct Win;

pub fn open() -> Res<Box<dyn Desktop>> {
    Ok(Box::new(Win))
}

fn hwnd(id: u64) -> HWND {
    id as usize as HWND
}

fn text(h: HWND) -> String {
    unsafe {
        let n = GetWindowTextLengthW(h);
        if n <= 0 {
            return String::new();
        }
        let mut buf = vec![0u16; n as usize + 1];
        let got = GetWindowTextW(h, buf.as_mut_ptr(), buf.len() as i32);
        String::from_utf16_lossy(&buf[..got.max(0) as usize])
    }
}

fn process_name(pid: u32) -> String {
    unsafe {
        let h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if h.is_null() {
            return String::new();
        }
        let mut buf = vec![0u16; 1024];
        let mut len = buf.len() as u32;
        let ok = QueryFullProcessImageNameW(h, PROCESS_NAME_WIN32, buf.as_mut_ptr(), &mut len);
        CloseHandle(h);
        if ok == 0 {
            return String::new();
        }
        let path = String::from_utf16_lossy(&buf[..len as usize]);
        path.rsplit(['\\', '/']).next().unwrap_or("").to_string()
    }
}

unsafe extern "system" fn collect(h: HWND, all: LPARAM) -> windows_sys::core::BOOL {
    (*(all as *mut Vec<HWND>)).push(h);
    1
}

fn client_size(h: HWND) -> (u32, u32) {
    let mut r = RECT { left: 0, top: 0, right: 0, bottom: 0 };
    unsafe { GetClientRect(h, &mut r) };
    ((r.right - r.left).max(0) as u32, (r.bottom - r.top).max(0) as u32)
}

fn send(input: INPUT) -> Res<()> {
    let sent = unsafe { SendInput(1, &input, std::mem::size_of::<INPUT>() as i32) };
    if sent == 1 {
        Ok(())
    } else {
        Err("Windows refused the input (another program may be blocking input)".into())
    }
}

impl Desktop for Win {
    fn windows(&mut self) -> Res<Vec<WindowInfo>> {
        let mut all: Vec<HWND> = Vec::new();
        unsafe { EnumWindows(Some(collect), &mut all as *mut Vec<HWND> as LPARAM) };
        let me = std::process::id();
        let mut out = Vec::new();
        for h in all {
            unsafe {
                if IsWindowVisible(h) == 0 || !GetWindow(h, GW_OWNER).is_null() {
                    continue;
                }
                if GetWindowLongPtrW(h, GWL_EXSTYLE) as u32 & WS_EX_TOOLWINDOW != 0 {
                    continue;
                }
            }
            let title = text(h);
            let mut pid = 0u32;
            unsafe { GetWindowThreadProcessId(h, &mut pid) };
            if title.is_empty() || pid == me {
                continue;
            }
            let (width, height) = client_size(h);
            if width == 0 || height == 0 {
                continue;
            }
            out.push(WindowInfo {
                id: h as usize as u64,
                title,
                process: process_name(pid),
                pid,
                width,
                height,
            });
        }
        Ok(out)
    }

    fn in_front(&mut self, window: u64) -> bool {
        unsafe { GetForegroundWindow() == hwnd(window) }
    }

    fn bring_to_front(&mut self, window: u64) -> Res<()> {
        let h = hwnd(window);
        unsafe {
            if IsWindow(h) == 0 {
                return Err("The window is gone".into());
            }
            if IsIconic(h) != 0 {
                ShowWindow(h, SW_RESTORE);
            }
            if SetForegroundWindow(h) == 0 {
                return Err("Windows did not let Tools bring the game to the front: click on the game".into());
            }
        }
        Ok(())
    }

    fn area(&mut self, window: u64) -> Res<Rect> {
        let h = hwnd(window);
        if unsafe { IsWindow(h) } == 0 {
            return Err("The window is gone".into());
        }
        let (width, height) = client_size(h);
        let mut p = POINT { x: 0, y: 0 };
        unsafe { ClientToScreen(h, &mut p) };
        Ok(Rect { x: p.x, y: p.y, width, height })
    }

    fn key(&mut self, key: &KeyInfo, down: bool) -> Res<()> {
        let mut flags = KEYEVENTF_SCANCODE;
        if key.extended {
            flags |= KEYEVENTF_EXTENDEDKEY;
        }
        if !down {
            flags |= KEYEVENTF_KEYUP;
        }
        send(INPUT {
            r#type: INPUT_KEYBOARD,
            Anonymous: INPUT_0 {
                ki: KEYBDINPUT { wVk: 0, wScan: key.scan, dwFlags: flags, time: 0, dwExtraInfo: MARK },
            },
        })
    }

    fn button(&mut self, button: Button, down: bool) -> Res<()> {
        let flags = match (button, down) {
            (Button::Left, true) => MOUSEEVENTF_LEFTDOWN,
            (Button::Left, false) => MOUSEEVENTF_LEFTUP,
            (Button::Right, true) => MOUSEEVENTF_RIGHTDOWN,
            (Button::Right, false) => MOUSEEVENTF_RIGHTUP,
            (Button::Middle, true) => MOUSEEVENTF_MIDDLEDOWN,
            (Button::Middle, false) => MOUSEEVENTF_MIDDLEUP,
        };
        send(mouse(0, 0, flags))
    }

    fn motion(&mut self, dx: i32, dy: i32) -> Res<()> {
        send(mouse(dx, dy, MOUSEEVENTF_MOVE))
    }

    fn grab(&mut self, window: u64, width: u32, height: u32) -> Res<Vec<u8>> {
        let area = self.area(window)?;
        let (w, h) = (width as i32, height as i32);
        unsafe {
            let screen = GetDC(null_mut());
            if screen.is_null() {
                return Err("Windows did not give Tools the screen".into());
            }
            let mem = CreateCompatibleDC(screen);
            let mut info: BITMAPINFO = std::mem::zeroed();
            info.bmiHeader = BITMAPINFOHEADER {
                biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
                biWidth: w,
                biHeight: -h,
                biPlanes: 1,
                biBitCount: 32,
                biCompression: BI_RGB,
                ..std::mem::zeroed()
            };
            let mut bits: *mut core::ffi::c_void = null_mut();
            let dib = CreateDIBSection(mem, &info, DIB_RGB_COLORS, &mut bits, null_mut(), 0);
            let result = if dib.is_null() || bits.is_null() {
                Err("Windows could not make a picture".to_string())
            } else {
                let old = SelectObject(mem, dib);
                SetStretchBltMode(mem, HALFTONE);
                SetBrushOrgEx(mem, 0, 0, null_mut());
                let ok = StretchBlt(
                    mem,
                    0,
                    0,
                    w,
                    h,
                    screen,
                    area.x,
                    area.y,
                    area.width as i32,
                    area.height as i32,
                    SRCCOPY,
                );
                let out = if ok == 0 {
                    Err("Windows could not copy the game's picture".to_string())
                } else {
                    let px = std::slice::from_raw_parts(bits as *const u8, (w * h * 4) as usize);
                    // Blue, green, red, unused.
                    Ok(px.as_chunks::<4>().0.iter().flat_map(|p| [p[2], p[1], p[0]]).collect())
                };
                SelectObject(mem, old);
                DeleteObject(dib);
                out
            };
            DeleteDC(mem);
            ReleaseDC(null_mut(), screen);
            result
        }
    }
}

fn mouse(dx: i32, dy: i32, flags: u32) -> INPUT {
    INPUT {
        r#type: INPUT_MOUSE,
        Anonymous: INPUT_0 {
            mi: MOUSEINPUT { dx, dy, mouseData: 0, dwFlags: flags, time: 0, dwExtraInfo: MARK },
        },
    }
}

static STOP_SIGNAL: OnceLock<mpsc::Sender<()>> = OnceLock::new();

unsafe extern "system" fn keyboard_hook(code: i32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    if code == HC_ACTION as i32 && (wparam == WM_KEYDOWN as WPARAM || wparam == WM_SYSKEYDOWN as WPARAM) {
        let kb = &*(lparam as *const KBDLLHOOKSTRUCT);
        let held = |vk: u16| GetAsyncKeyState(vk as i32) as u16 & 0x8000 != 0;
        if kb.vkCode == VK_END as u32 && held(VK_CONTROL) && held(VK_MENU) {
            if let Some(tx) = STOP_SIGNAL.get() {
                let _ = tx.send(());
            }
            // The game never sees the stop keys.
            return 1;
        }
    }
    CallNextHookEx(null_mut(), code, wparam, lparam)
}

/// Watches for the emergency stop keys (Ctrl+Alt+End) in every program, with a low-level
/// keyboard hook on its own thread; `on_stop` runs on another thread so the hook stays quick.
pub fn stop_key_watch(on_stop: Box<dyn Fn() + Send>) -> Res<()> {
    let (tx, rx) = mpsc::channel();
    if STOP_SIGNAL.set(tx).is_err() {
        return Err("The emergency stop keys are already watched".into());
    }
    std::thread::Builder::new()
        .name("emergency stop".into())
        .spawn(move || {
            for () in rx {
                on_stop();
            }
        })
        .map_err(|e| e.to_string())?;
    let (ready_tx, ready_rx) = mpsc::channel();
    std::thread::Builder::new()
        .name("emergency stop keys".into())
        .spawn(move || unsafe {
            let hook =
                SetWindowsHookExW(WH_KEYBOARD_LL, Some(keyboard_hook), GetModuleHandleW(std::ptr::null()), 0);
            let _ = ready_tx.send(!hook.is_null());
            if hook.is_null() {
                return;
            }
            let mut msg: MSG = std::mem::zeroed();
            while GetMessageW(&mut msg, null_mut(), 0, 0) > 0 {}
        })
        .map_err(|e| e.to_string())?;
    match ready_rx.recv() {
        Ok(true) => Ok(()),
        _ => Err("Windows did not let Tools watch the emergency stop keys".into()),
    }
}

/// Listens to the keyboard and mouse through Raw Input on a hidden window (what the devices
/// report, whichever program is in front): keys from the controls table, the three buttons,
/// and relative movement. Injected input (from programs, Tools included) has no device and is
/// skipped unless `accept_virtual`, which only tests use.
///
/// Raw Input is registered once per process for each kind of device, so this replaces the
/// registration of Tools' windows while listening (they don't use raw input).
pub fn listen(accept_virtual: bool, tx: mpsc::Sender<super::Heard>) -> Res<super::Listening> {
    use super::Input;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Instant;
    use windows_sys::Win32::UI::Input::MOUSE_MOVE_ABSOLUTE;
    use windows_sys::Win32::UI::Input::{
        GetRawInputData, RegisterRawInputDevices, HRAWINPUT, RAWINPUT, RAWINPUTDEVICE, RAWINPUTHEADER,
        RIDEV_INPUTSINK, RIDEV_REMOVE, RID_INPUT, RIM_TYPEKEYBOARD, RIM_TYPEMOUSE,
    };
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        CreateWindowExW, DefWindowProcW, DestroyWindow, DispatchMessageW, PostMessageW, RegisterClassW,
        HWND_MESSAGE, RI_KEY_BREAK, RI_KEY_E0, RI_MOUSE_LEFT_BUTTON_DOWN, RI_MOUSE_LEFT_BUTTON_UP,
        RI_MOUSE_MIDDLE_BUTTON_DOWN, RI_MOUSE_MIDDLE_BUTTON_UP, RI_MOUSE_RIGHT_BUTTON_DOWN,
        RI_MOUSE_RIGHT_BUTTON_UP, WM_INPUT, WM_USER, WNDCLASSW,
    };

    unsafe extern "system" fn proc(h: HWND, m: u32, w: WPARAM, l: LPARAM) -> LRESULT {
        DefWindowProcW(h, m, w, l)
    }
    fn key_name(scan: u16, e0: bool) -> Option<&'static str> {
        match (scan, e0) {
            (0x36, false) => return Some("shift"),
            (0x1D, true) => return Some("ctrl"),
            (0x38, true) => return Some("alt"),
            (0x1C, true) => return Some("enter"),
            // Shift codes Windows adds around some keys, not keys pressed.
            (0x2A, true) | (0x36, true) => return None,
            _ => {}
        }
        crate::live::controls::KEYS.iter().find(|k| k.scan == scan && k.extended == e0).map(|k| k.name)
    }

    let stop = Arc::new(AtomicBool::new(false));
    let window = Arc::new(AtomicUsize::new(0));
    let (ready_tx, ready_rx) = mpsc::channel::<Res<()>>();
    let (stopping, win) = (stop.clone(), window.clone());
    const WAKE: u32 = WM_USER + 7;
    let thread = std::thread::Builder::new()
        .name("input listener".into())
        .spawn(move || unsafe {
            let class: Vec<u16> = "YendoryToolsInput\0".encode_utf16().collect();
            let instance = GetModuleHandleW(std::ptr::null());
            let wc = WNDCLASSW {
                lpfnWndProc: Some(proc),
                hInstance: instance,
                lpszClassName: class.as_ptr(),
                ..std::mem::zeroed()
            };
            RegisterClassW(&wc); // Fails harmlessly when already registered by an earlier recording.
            let hwnd = CreateWindowExW(
                0,
                class.as_ptr(),
                std::ptr::null(),
                0,
                0,
                0,
                0,
                0,
                HWND_MESSAGE,
                null_mut(),
                instance,
                std::ptr::null(),
            );
            if hwnd.is_null() {
                let _ =
                    ready_tx.send(Err("Windows did not let Tools listen to the keyboard and mouse".into()));
                return;
            }
            let devices = [
                RAWINPUTDEVICE { usUsagePage: 1, usUsage: 6, dwFlags: RIDEV_INPUTSINK, hwndTarget: hwnd },
                RAWINPUTDEVICE { usUsagePage: 1, usUsage: 2, dwFlags: RIDEV_INPUTSINK, hwndTarget: hwnd },
            ];
            if RegisterRawInputDevices(devices.as_ptr(), 2, std::mem::size_of::<RAWINPUTDEVICE>() as u32) == 0
            {
                DestroyWindow(hwnd);
                let _ =
                    ready_tx.send(Err("Windows did not let Tools listen to the keyboard and mouse".into()));
                return;
            }
            win.store(hwnd as usize, Ordering::Release);
            let _ = ready_tx.send(Ok(()));
            let mut msg: MSG = std::mem::zeroed();
            // u64s, so the RAWINPUT read from it is aligned.
            let mut buf = vec![0u64; 128];
            while !stopping.load(Ordering::Acquire) && GetMessageW(&mut msg, null_mut(), 0, 0) > 0 {
                if msg.message == WAKE {
                    continue;
                }
                if msg.message == WM_INPUT {
                    let now = Instant::now();
                    let mut size = (buf.len() * 8) as u32;
                    let got = GetRawInputData(
                        msg.lParam as HRAWINPUT,
                        RID_INPUT,
                        buf.as_mut_ptr() as *mut _,
                        &mut size,
                        std::mem::size_of::<RAWINPUTHEADER>() as u32,
                    );
                    if got != u32::MAX && got as usize >= std::mem::size_of::<RAWINPUTHEADER>() {
                        let raw = &*(buf.as_ptr() as *const RAWINPUT);
                        let injected = raw.header.hDevice.is_null();
                        let mut heard: Vec<Input> = Vec::new();
                        if !injected || accept_virtual {
                            if raw.header.dwType == RIM_TYPEKEYBOARD {
                                let k = raw.data.keyboard;
                                let e0 = k.Flags as u32 & RI_KEY_E0 != 0;
                                if let Some(key) = key_name(k.MakeCode, e0) {
                                    heard.push(Input::Key { key, down: k.Flags as u32 & RI_KEY_BREAK == 0 });
                                }
                            } else if raw.header.dwType == RIM_TYPEMOUSE {
                                let m = raw.data.mouse;
                                if m.usFlags & MOUSE_MOVE_ABSOLUTE == 0 && (m.lLastX != 0 || m.lLastY != 0) {
                                    heard.push(Input::Motion { dx: m.lLastX as f64, dy: m.lLastY as f64 });
                                }
                                let flags = m.Anonymous.Anonymous.usButtonFlags as u32;
                                for (bit, button, down) in [
                                    (RI_MOUSE_LEFT_BUTTON_DOWN, Button::Left, true),
                                    (RI_MOUSE_LEFT_BUTTON_UP, Button::Left, false),
                                    (RI_MOUSE_RIGHT_BUTTON_DOWN, Button::Right, true),
                                    (RI_MOUSE_RIGHT_BUTTON_UP, Button::Right, false),
                                    (RI_MOUSE_MIDDLE_BUTTON_DOWN, Button::Middle, true),
                                    (RI_MOUSE_MIDDLE_BUTTON_UP, Button::Middle, false),
                                ] {
                                    if flags & bit != 0 {
                                        heard.push(Input::Button { button, down });
                                    }
                                }
                            }
                        }
                        for h in heard {
                            if tx.send((now, h)).is_err() {
                                stopping.store(true, Ordering::Release);
                            }
                        }
                    }
                }
                DispatchMessageW(&msg);
            }
            let remove = [
                RAWINPUTDEVICE { usUsagePage: 1, usUsage: 6, dwFlags: RIDEV_REMOVE, hwndTarget: null_mut() },
                RAWINPUTDEVICE { usUsagePage: 1, usUsage: 2, dwFlags: RIDEV_REMOVE, hwndTarget: null_mut() },
            ];
            RegisterRawInputDevices(remove.as_ptr(), 2, std::mem::size_of::<RAWINPUTDEVICE>() as u32);
            DestroyWindow(hwnd);
        })
        .map_err(|e| e.to_string())?;
    match ready_rx.recv() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => return Err(e),
        Err(_) => return Err("Windows did not let Tools listen to the keyboard and mouse".into()),
    }
    let wake: Box<dyn Fn() + Send> = Box::new(move || unsafe {
        PostMessageW(window.load(Ordering::Acquire) as HWND, WAKE, 0, 0);
    });
    Ok(super::Listening::new(stop, Some(wake), thread))
}
