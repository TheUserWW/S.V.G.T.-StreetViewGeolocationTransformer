import ctypes
import json
import os
from pathlib import Path
import tempfile
import time
import tkinter as tk
import unittest
from unittest.mock import patch

from PIL import Image

from streetgeo.client.workbench import DailyApp
from streetgeo.core.checkpoints import save_checkpoint, read_checkpoint
from streetgeo.core.model import StreetGeoModel
from streetgeo.inference.sessions import SessionStore
from streetgeo.training.online import learn_from_feedback
from test_model import tiny_config


class SessionTests(unittest.TestCase):
    def test_history_is_bounded_reloads_and_only_prunes_owned_captures(self):
        with tempfile.TemporaryDirectory() as folder:
            store=SessionStore(Path(folder)/'workspace',limit=2)
            source=Path(folder)/'original.png';Image.new('RGB',(8,8)).save(source)
            captures=[]
            for i in range(3):
                captures.append(store.capture(Image.new('RGB',(8,8))))
                store.save({'images':[captures[-1],str(source)],'candidates':[], 'round_id':str(i)})
            store.prune_captures()
            self.assertFalse(Path(captures[0]).exists())
            self.assertTrue(source.exists())
            self.assertTrue(Path(captures[1]).exists())
            self.assertEqual([r['round_id'] for r in SessionStore(store.directory,limit=2).records],['1','2'])

    def test_corrupt_history_does_not_prevent_startup(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder)/'history.json').write_text('not json')
            self.assertEqual(SessionStore(folder).records,[])

    def test_personal_checkpoint_can_continue_after_moving(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);base=root/'base.pt';image=root/'image.png'
            save_checkpoint(StreetGeoModel(tiny_config()),base)
            Image.new('RGB',(32,32),'orange').save(image)
            first=learn_from_feedback(base,[image],38,-123,device='cpu',output_root=root/'learning')
            moved=root/'moved.pt';Path(first['checkpoint']).rename(moved)
            next_result=learn_from_feedback(moved,[image],38.1,-123,device='cpu')
            self.assertEqual(next_result['updates'],2)
            self.assertEqual(read_checkpoint(moved)['training_state']['online_learning']['personal_checkpoint'],str(moved))


class DailyWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.folder=Path(self.temp.name)
        self.model=self.folder/'V4.pt';self.image=self.folder/'街景 测试.png'
        save_checkpoint(StreetGeoModel(tiny_config()).eval(),self.model)
        Image.new('RGB',(120,80),'green').save(self.image)
        self.root=tk.Tk();self.root.withdraw()
        self.app=DailyApp(self.root,settings_path=self.folder/'settings.json',checkpoint=self.model,language='zh',device='cpu')
        self.errors=[];self.app.error=self.errors.append
        self.root.update_idletasks()

    def tearDown(self):
        self.app.close();self.temp.cleanup()

    def wait(self):
        end=time.monotonic()+20
        while self.app.busy and time.monotonic()<end:
            self.root.update();time.sleep(.01)
        self.assertFalse(self.app.busy)
        self.assertFalse(self.errors,self.errors)

    def predict(self):
        self.app.set_images([self.image]);self.app.start_prediction();self.wait()

    def test_paste_automatically_predicts_and_new_round_clears_old_pin(self):
        with patch('streetgeo.client.workbench.clipboard_content',return_value=Image.new('RGB',(100,60),'blue')):
            self.assertEqual(self.app.paste_images(),'break')
            self.assertTrue(self.app.busy)
            self.wait()
            self.assertTrue(Path(self.app.images[0]).is_relative_to(self.app.store.captures))
            self.app.map_click(38.04,-122.98)
            self.assertIsNotNone(self.app.manual_point)
            self.assertFalse(self.app.busy)  # Manual pinning never learns by default.
            self.app.paste_images();self.wait()
        self.assertIsNone(self.app.manual_point)
        self.assertEqual(len(self.app.store.records),2)

    def test_manual_coordinates_language_switch_and_history_reopen(self):
        self.predict();self.app.pin_text.set('38.04, -122.98');self.app.apply_pin()
        record_id=self.app.result['round_id'];position=self.app.map.center
        self.app.open_history();self.app.open_settings()
        self.app.language_choice.set('English');self.app.change_language()
        self.assertEqual(self.app.map.center,position)
        self.assertEqual(self.app.map.ground_truth['longitude'],-122.98)
        self.assertEqual(self.app.learn_button.cget('text'),'Learn this correction')
        self.assertEqual(self.app.history_table.heading('status')['text'],'Status')
        self.app.clear_images();self.app.open_round(record_id)
        self.assertEqual(self.app.manual_point['latitude'],38.04)
        self.assertEqual(self.app.result['round_id'],record_id)
        self.assertIn('original predictions',self.app.status.get())

    def test_map_feedback_learns_and_profiles_switch_without_overwriting_base(self):
        self.predict();before=self.model.read_bytes()
        self.app.map_click(38.04,-122.98)
        self.assertEqual(self.app.learn_button.cget('state'),'normal')
        self.app.start_learning();self.wait()
        personal=self.app.personal_checkpoint
        self.assertNotEqual(str(personal),str(self.model))
        self.assertEqual(self.model.read_bytes(),before)
        self.assertTrue(self.app.result['learned'])
        self.app.profile_choice.set(self.app.t('base_model'));self.app.change_profile()
        self.assertEqual(self.app.checkpoint.get(),str(self.model))
        self.app.profile_choice.set(self.app.t('personal_model'));self.app.change_profile()
        self.assertEqual(self.app.checkpoint.get(),personal)
        self.app.undo_learning();self.wait()
        self.assertEqual(self.app.checkpoint.get(),str(self.model))
        self.assertFalse(self.app.personal_checkpoint)

    def test_explicit_auto_learning_mode_and_no_self_labels(self):
        self.predict();self.app.auto_learn.set(True)
        self.app.map_click(35,139)
        self.assertTrue(self.app.busy);self.wait()
        state=read_checkpoint(self.app.checkpoint.get())['training_state']['online_learning']
        self.assertEqual(state['memory'][0]['coordinates'],[35.0,139.0])

    def test_coordinate_entry_keeps_standard_text_paste(self):
        with patch.object(self.root,'focus_get',return_value=self.app.pin_entry), \
                patch('streetgeo.client.workbench.clipboard_content') as grab:
            self.assertIsNone(self.app.paste_images(object()))
            grab.assert_not_called()

    @unittest.skipUnless(os.name=='nt','Windows native drops')
    def test_native_unicode_file_drop_reaches_image_import(self):
        drop=self.app.drop_target
        self.assertTrue(drop and drop.enabled)
        self.app.auto_predict.set(False)
        from ctypes import wintypes
        class DropFiles(ctypes.Structure):
            _fields_=[('pFiles',wintypes.DWORD),('pt',wintypes.POINT),('fNC',wintypes.BOOL),('fWide',wintypes.BOOL)]
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.GlobalAlloc.argtypes=[wintypes.UINT,ctypes.c_size_t];kernel.GlobalAlloc.restype=ctypes.c_void_p
        kernel.GlobalLock.argtypes=[ctypes.c_void_p];kernel.GlobalLock.restype=ctypes.c_void_p
        kernel.GlobalUnlock.argtypes=[ctypes.c_void_p]
        payload=(str(self.image)+'\0\0').encode('utf-16-le')
        header=DropFiles(ctypes.sizeof(DropFiles),wintypes.POINT(0,0),False,True)
        handle=kernel.GlobalAlloc(2,ctypes.sizeof(header)+len(payload));ptr=kernel.GlobalLock(handle)
        ctypes.memmove(ptr,ctypes.byref(header),ctypes.sizeof(header))
        ctypes.memmove(ptr+ctypes.sizeof(header),payload,len(payload));kernel.GlobalUnlock(handle)
        user=ctypes.WinDLL('user32');user.PostMessageW.argtypes=[wintypes.HWND,wintypes.UINT,wintypes.WPARAM,wintypes.LPARAM]
        self.assertTrue(user.PostMessageW(drop.hwnd,0x233,handle,0))
        end=time.monotonic()+3
        while not self.app.images and time.monotonic()<end:self.root.update();time.sleep(.01)
        self.assertEqual(self.app.images,[str(self.image)])


if __name__=='__main__':unittest.main()
