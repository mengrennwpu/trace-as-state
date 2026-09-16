from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import hashlib


def append_jsonl(path: str | Path, obj: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def md5(text):
    m = hashlib.md5()
    m.update(text.encode('utf-8'))
    return m.hexdigest()


def get_datas(file_path, json_flag=True, all_flag=False, mode='r'):
    """读取文本文件"""
    results = []
    
    with open(file_path, mode, encoding='utf-8') as f:
        for line in f.readlines():
            if not all_flag and json_flag:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
            else:
                results.append(line.strip())
        if all_flag:
            if json_flag:
                return json.loads(''.join(results))
            else:
                return '\n'.join(results)
        return results
    

def save_datas(file_path, datas, json_flag=True, all_flag=False, with_indent=False, mode='w'):
    """保存文本文件"""
    with open(file_path, mode, encoding='utf-8') as f:
        if all_flag:
            if json_flag:
                f.write(json.dumps(datas, ensure_ascii=False, indent= 4 if with_indent else None))
            else:
                f.write(''.join(datas))
        else:
            for data in datas:
                if json_flag:
                    f.write(json.dumps(data, ensure_ascii=False) + '\n') 
                else:
                    f.write(data + '\n')