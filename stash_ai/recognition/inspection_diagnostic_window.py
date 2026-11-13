"""On-demand diagnostics, kept separate from routine image/navigation work."""
from functools import partial
import tkinter as tk
from tkinter import ttk

from .inspection_runtime import Worker
from .inspection_diagnostics import stages, contribution, retained_rivals, review_queue, unassigned_summary
from .inspection_candidates import calculate
from .inspection_context import locate


class Diagnostics:
    def build_diagnostics(self,right):
        self.detail_data=None
        bar=ttk.Frame(right);bar.grid(row=7,column=0,sticky='ew')
        self.decision_button=ttk.Button(bar,text='Decision details',command=self.decision_details)
        self.decision_button.pack(side='left');self.decision_button.state(['disabled'])
        if hasattr(self.reader,'catalogue'):
            ttk.Button(bar,text='Review queue',command=lambda:self.open_diagnostics('review')).pack(side='left')
            ttk.Button(bar,text='Unassigned summary',command=lambda:self.open_diagnostics('summary')).pack(side='left')

    def decision_details(self):
        if self.detail_data and self.current_row:self.open_diagnostics('decision')

    def open_diagnostics(self,mode):
        window=DiagnosticWindow(self,mode);self.visual_windows.append(window)


class DiagnosticWindow:
    def __init__(self,host,mode):
        self.host,self.mode=host,mode
        self.reader=host.reader;self.detail=host.detail_data
        self.row=dict(host.current_row) if host.current_row else None
        self.scope=host.preview_scope
        self.closed=False;self.actions={};self.review_rows=[];self.page=0
        self.window=tk.Toplevel(host.root);self.window.title({'decision':'Decision details','review':'Review queue','summary':'Unassigned summary'}[mode])
        self.window.geometry('1050x650')
        self.worker=Worker('inspector-diagnostics')
        bar=ttk.Frame(self.window,padding=6);bar.pack(fill='x')
        self.run=ttk.Button(bar,text='Calculate candidates' if mode=='decision' else 'Refresh',command=self.load)
        self.run.pack(side='left')
        ttk.Button(bar,text='Cancel',command=self.cancel).pack(side='left')
        self.status=ttk.Label(bar,text='',wraplength=850);self.status.pack(side='left',padx=8)
        self.tabs=ttk.Notebook(self.window);self.tabs.pack(fill='both',expand=True)
        self.info=tk.Text(self.window,height=5,wrap='word');self.info.pack(fill='x')
        self.window.protocol('WM_DELETE_WINDOW',self.close)
        self.timer=self.window.after(30,self.poll)
        if mode=='decision':self.show_decision()
        else:self.load()

    def note(self,text):
        self.info.configure(state='normal');self.info.delete('1.0','end');self.info.insert('1.0',text);self.info.configure(state='disabled')

    def tree(self,title,columns):
        frame=ttk.Frame(self.tabs);self.tabs.add(frame,text=title)
        frame.rowconfigure(0,weight=1);frame.columnconfigure(0,weight=1)
        tree=ttk.Treeview(frame,columns=columns,show='tree headings')
        tree.heading('#0',text='Stage / observation');tree.column('#0',width=280)
        for column in columns:tree.heading(column,text=column.replace('_',' ').capitalize());tree.column(column,width=120)
        tree.grid(row=0,column=0,sticky='nsew')
        scroll=ttk.Scrollbar(frame,command=tree.yview);scroll.grid(row=0,column=1,sticky='ns');tree.configure(yscrollcommand=scroll.set)
        horizontal=ttk.Scrollbar(frame,orient='horizontal',command=tree.xview);horizontal.grid(row=1,column=0,sticky='ew');tree.configure(xscrollcommand=horizontal.set)
        tree.bind('<Double-Button-1>',lambda _e:self.activate(tree))
        for tag,color in [('fail','#b42318'),('unknown','#975b00'),('pass','#167044')]:tree.tag_configure(tag,foreground=color)
        return tree

    def action(self,tree,item,callback):self.actions[tree,item]=callback

    def activate(self,tree):
        items=tree.selection();action=self.actions.get((tree,items[0])) if items else None
        if action:action()

    def navigate(self,kind,address):
        self.finished('Opening evidence…')
        self.worker.request(partial(locate,self.reader.config,kind,address,self.scope),self.host.open_row,self.error)

    def show_decision(self):
        tree=self.tree('Decision chain',('status','value','source'))
        for stage in stages(self.detail):
            item=tree.insert('','end',text=stage['label'],values=(stage['status'],stage['note'],stage['source']),tags=(stage['status'],),open=True)
            for check in stage['children']:
                child=tree.insert(item,'end',text=check['label'],values=(check['status'],check['value'],check['note']),tags=(check['status'],))
                if 'reference' in check:
                    ref=check['reference'];self.action(tree,child,partial(self.navigate,ref['kind'],ref['observation']))
        rivals=self.tree('Recorded rivals',('identity','score','policy'))
        records=retained_rivals(self.detail)
        for record in records:rivals.insert('','end',text=record['route'],values=(record['identity'] or 'Not retained',record['score'] if record['score'] is not None else 'Unknown',record['policy']))
        if not records:rivals.insert('','end',text='No competitor result retained. Use Calculate candidates for a new comparison.')
        self.candidates=self.tree('New comparison',('performer','mean','minimum','maximum','references','bank_selection'))
        self.note(contribution(self.detail))
        self.status.configure(text='Retained decision evidence. Double-click an evidence ID to inspect it.')

    def load(self):
        self.run.state(['disabled']);self.status.configure(text='Calculating… close or Cancel to stop.')
        if self.mode=='decision':
            task=partial(calculate,self.reader.config,self.row['kind'],self.row['id'],self.scope);callback=self.show_candidates
        elif self.mode=='review':task=partial(review_queue,self.reader);callback=self.show_review
        else:task=partial(unassigned_summary,self.reader);callback=self.show_summary
        self.worker.request(task,callback,self.error)

    def finished(self,text,changed=False):
        self.run.state(['!disabled']);self.status.configure(text=text+(' · Database changed during calculation; refresh.' if changed else ''))

    def show_candidates(self,result):
        tree=self.candidates
        self.actions={k:v for k,v in self.actions.items() if k[0]!=tree}
        tree.delete(*tree.get_children())
        for row in result['rows']:
            value=lambda k:f'{row[k]:.4f}' if row[k] is not None else 'Unknown'
            item=tree.insert('','end',text=('Current owner · ' if row['owner'] else '')+row['identity'],
                values=(row['performer'] or 'Internal identity',value('score'),value('minimum'),value('maximum'),row['count'],
                        ('Query selected' if row['selected'] is True else 'Query not selected' if row['selected'] is False else 'No eligible bank')+' in this comparison'))
            if row['witness']:self.action(tree,item,partial(self.navigate,result['kind'],row['witness']))
        if not result['rows']:tree.insert('','end',text='No comparable retained references.')
        self.tabs.select(tree.master)
        self.note(result['note']+'\nDouble-click a candidate to inspect its highest-scoring reference. Pin it for visual comparison.')
        self.finished(f"{result['compared']} identities examined · {result['invalid']} invalid vectors omitted"+
                      (' · Budget limited result' if result['truncated'] else ''),result['changed'])

    def clear_tabs(self):
        for tab in self.tabs.tabs():self.tabs.nametowidget(tab).destroy()
        self.actions.clear()

    def show_review(self,result):
        self.clear_tabs();self.review_rows=result['rows'];self.page=0
        self.review_tree=self.tree('Review priority',('priority','scene','cue','kind','reasons'))
        bar=ttk.Frame(self.review_tree.master);bar.grid(row=2,column=0,sticky='ew')
        self.previous=ttk.Button(bar,text='Previous',command=lambda:self.move(-1));self.previous.pack(side='left')
        self.next=ttk.Button(bar,text='Next',command=lambda:self.move(1));self.next.pack(side='left')
        self.render_review()
        self.finished(f"{result['seen']} accepted samples checked · {result['flagged']} flagged · top {len(result['rows'])} retained",result['changed'])
        self.note('Review priority is a transparent sum of heuristics, not an error probability: retained rejection +5, lead ≤0.03 +4, '
                  'face sharpness below current role floor +3, historical/missing decision +2, unknown role +1. '
                  'It does not automatically detach, blacklist or alter any sample. Double-click a row to inspect.')

    def move(self,direction):self.page=max(0,self.page+direction);self.render_review()

    def render_review(self):
        tree=self.review_tree;tree.delete(*tree.get_children());self.actions.clear()
        for row in self.review_rows[self.page*80:(self.page+1)*80]:
            item=tree.insert('','end',text=row['id'],values=(row['priority'],row['scene'],row['cue'],row['kind'],'; '.join(row['reasons'])))
            self.action(tree,item,partial(self.host.open_row,row))
        self.previous.state(['!disabled'] if self.page else ['disabled'])
        self.next.state(['!disabled'] if (self.page+1)*80<len(self.review_rows) else ['disabled'])

    def show_summary(self,result):
        self.clear_tabs()
        tree=self.tree('Retained states',('kind','count'))
        for row in result['states']:
            item=tree.insert('','end',text=row['state'],values=(row['kind'],row['count']))
            self.action(tree,item,partial(self.filter_state,row))
        gates=self.tree('Quality gates',('kind','stage','failed','unknown','passed','threshold','measured_range','nearest_failure','source'))
        for row in result['gates']:
            known=row['minimum'] is not None
            gates.insert('','end',text=row['parameter'],values=(row['kind'],row['stage'],row['counts']['fail'],row['counts']['unknown'],row['counts']['pass'],
                f"{row['operator']} {row['threshold']}",f"{row['minimum']:g}…{row['maximum']:g}" if known else 'Unknown',
                f"{row['closest_failure']:g}" if row['closest_failure'] is not None else 'Unknown',row['source']))
        self.note('Counts cover unassigned observations from the entry scene. Gate failures can overlap; anchor and auxiliary gates are separate routes. '
                  'Measured ranges and nearest failed boundary use retained measurements, not a rerun. The source column distinguishes historical and current settings. '
                  'Double-click a state to filter the main view.')
        self.finished(f"{result['total']} unassigned observations examined",result['changed'])

    def filter_state(self,row):
        host=self.host;host.view_filters=dict(scene=self.reader.scene,kind=row['kind'],state=row['state'],assignment='unassigned')
        host.scene_entry.delete(0,'end');host.scene_entry.insert(0,self.reader.scene)
        host.restore_state=None;host.restoring=True;host.search();host.restoring=False

    def cancel(self):self.worker.invalidate();self.finished('Cancelled. Previous results remain visible.')
    def error(self,error):self.finished(str(error))
    def poll(self):
        if not self.closed:self.worker.poll();self.timer=self.window.after(30,self.poll)
    def close(self):
        if self.closed:return
        self.closed=True;self.window.after_cancel(self.timer);self.worker.close();self.window.destroy()
        self.actions.clear();self.review_rows.clear();self.detail=None
        if self in self.host.visual_windows:self.host.visual_windows.remove(self)
