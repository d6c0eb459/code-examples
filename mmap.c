/**
 * Managed memory mapping.
 *
 * Allocates, then manages a block of memory. Data grows from the bottom,
 * while a 'free' list grows downwards from the top.
 *
 * Defrag is outside the scope of this function because the mmap has no clue
 * what data should be placed beside each other.
 *
 * This file forms part of a project to create vector animation software.
 *
 * Copyright Derek Cheung 2016 All rights reserved.
 */
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "mmap.h"

struct mmap* mmap_create(int num_entries, int entry_size) {
    struct mmap *mmap = malloc(sizeof(struct mmap));
    mmap_init(mmap, num_entries, entry_size);
    return mmap;
}

/**
 * Initialize an mmap
 */
void mmap_init(struct mmap *mmap, int num_entries, int entry_size) {
    assert(num_entries > 0 && entry_size > 0);

    mmap->entry_size = entry_size;

    int size = entry_size * num_entries;
    mmap->data_size = size;
    mmap->data = malloc(size);

    mmap->next_available = mmap->data;
    mmap->free_list_bottom = mmap->data + size;
}

/**
 * Free the mmap
 */
int mmap_free(struct mmap *mmap, void *addr) {

    if(addr < mmap->data || 
            addr > (mmap->next_available - mmap->entry_size)
        ) 
    {
        // woah, addr lies outside this mmap...!
        return 0;
    }

    // is the address that of the next available? because if so, accept 
    // without further conditions
    if(addr == mmap->next_available - mmap->entry_size) {
        mmap->next_available = mmap->next_available - mmap->entry_size;
        return 1;
    }

    // otherwise, we'll need to register this in the free list.

    // let candidate be the place in the free list where we'd like to store
    // this entry
    
    // need at least an int of space
    int *candidate = mmap->free_list_bottom - sizeof(int);

    // redundant check - grow() should always
    // leave enough room.
    //
    // this cannot overlap with any already-existing entries
    if((void *) candidate < mmap->next_available) {
        // in the special case of it overlapping the requested
        // address's space, we'll let it slide. otherwise, denied.
        //if((void *) candidate >= addr + mmap->entry_size) {
            return 0;
        //}
    }

    // otherwise, go ahead and write the offset of the addr down
    *candidate = addr - mmap->data;

    // and update the free list bottom
    mmap->free_list_bottom = candidate;

    return 1;

}

/**
 * Make the mmap larger
 */
void *mmap_grow(struct mmap *mmap) {

    // check to see if we have a sane mmap
    assert(mmap->free_list_bottom != NULL 
            && mmap->data != NULL 
            && mmap->data_size > 0);

    // check for free spot available
    if(mmap->free_list_bottom < (void *) (mmap->data + mmap->data_size)) {

        // take it

        int offset = *((int *)mmap->free_list_bottom);
        mmap->free_list_bottom = mmap->free_list_bottom + sizeof(int);

        return mmap->data + offset;

    } else {

        // check for mmap full
        //
        // mmap is full if a new entry will not fit into the data without
        // overlapping into a potential free-ing entry. this is to prevent
        // situations when the mmap is 'stuffed' and free operations fail due
        // to lack of space to record the free entry
        // 
        if((mmap->next_available + mmap->entry_size)
                > (mmap->free_list_bottom - sizeof(int))) {
            // deny
            return NULL;
        }

        // start growing from bottom
        void *output = mmap->next_available;
        mmap->next_available = mmap->next_available + mmap->entry_size;

        return output;
    }

}

/**
 * Save/restore functions
 */

/**
 * Writes entire structure out to the given buffer.
 */
void mmap_write(struct io_buffer *buffer, struct mmap *mmap) {
    // write down all properties
    io_write(buffer, mmap, sizeof(struct mmap));
    // write the data section
    io_write(buffer, mmap->data, mmap->data_size);
}

/**
 * Loads data from a buffer into an mmap.
 * @return pointer to mmap
 */
struct mmap_linker mmap_read(struct io_buffer *buffer, struct mmap *mmap) {

    // todo assert mmap data pointer is valid

    // read right into the struct
    io_read(buffer, mmap, sizeof(struct mmap));

    // alloc for data
    void *new_data = malloc(mmap->data_size);

    // save old information
    struct mmap_linker linker = {
        mmap->data, // bottom
        mmap->next_available - mmap->entry_size, // valid top
        new_data
    };

    // update new struct with new information
    mmap->data = new_data;

    // read the data in
    io_read(buffer, mmap->data, mmap->data_size);

    // correct references
    mmap->next_available = (mmap->next_available - linker.start) + mmap->data;
    mmap->free_list_bottom = (mmap->free_list_bottom - linker.start) 
        + mmap->data;
    
    return linker;
}

/**
 * Debuging and data visualization functions
 */

void mmap_stats(struct mmap *mmap) {
    printf("+-------------------------+\n");
    printf("| slots                   |\n");
    printf("+--------------+----------+\n");

    int num_free = (mmap->data + mmap->data_size - mmap->free_list_bottom)
        / sizeof(int);

    int slots = (mmap->next_available - mmap->data) / mmap->entry_size;

    int capacity = (mmap->data_size / mmap->entry_size) - 1;

    printf("| start        | %8x |\n", mmap->data);
    printf("| total        | %8i |\n", capacity);
    printf("| bytes per    | %8i |\n", mmap->entry_size);
    printf("| total bytes  | %8i |\n", capacity * mmap->entry_size);
    printf("| used         | %8i |\n", slots - num_free);
    printf("| free         | %8i |\n", num_free);
    printf("| used height  | %8i |\n", slots);
    printf("+--------------+----------+\n");
}

void mmap_dump(struct mmap *mmap) {
    
    // for debug, we just dump to console
    printf("+----------+-------------------------+\n");
    int i;
    int limit = mmap->data_size / (sizeof(char));
    char *data = mmap->data;
    for(i=0;i<limit;i++) {
        if(i % 8 == 0) {
            printf("| %8x | ", &data[i]);
        }
        printf("%02hhx ", data[i]);
        if(i % 8 == 7) {
            printf("|\n");
        }
    }
    if(i % 8 != 0) {
        printf("\n");
    }
    mmap_stats(mmap);

}

/**
 * List all free spots
 */
void mmap_list_free(struct mmap *mmap) {

    int i;
    int num_free = (mmap->data + mmap->data_size - mmap->free_list_bottom)
        / sizeof(int);

    int *byte_offset;

    printf("+------------+\n");
    printf("| free slots |\n");

    for(i=0;i<num_free;i++) {
        byte_offset =  mmap->free_list_bottom + (i * sizeof(int));
        printf("%p\n", mmap->data + *byte_offset);
    }

}
