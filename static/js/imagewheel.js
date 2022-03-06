$(function(){

  var position = null
  var size = null
  const regex = /-?[0-9]+/g;
  var is_down = false;
  var drag_start_X = 0;
  var drag_start_Y = 0;
  var mouse_move_counter = 0;

  function gposition(){
    var position = $("#img-div").css("background-position");
    position = position.match(regex);
    return position
  }

  function gsize(){
    var size = $("#img-div").css("background-size")
    size = size.match(regex);
    return size
  }

  function zoomin() {
      size = gsize();
      const newWidth = parseInt(parseInt(size[0])*1.1);
      const newHight = parseInt(parseInt(size[1])*1.1);
      $("#img-div").css("background-size", `${newWidth}px ${newHight}px`);
  }

  function zoomout() {
    size = gsize();
    const newWidth = parseInt(parseInt(size[0])*0.9);
    const newHight = parseInt(parseInt(size[1])*0.9);
    $("#img-div").css("background-size", `${newWidth}px ${newHight}px`);
  }

  function adjust_position(dx, dy){
    position = gposition()
    const newX = parseInt(position[0])+dx;
    const newY = parseInt(position[1])+dy;
    $("#img-div").css("background-position", `${newX}px ${newY}px, left top`)
  };

// drag controls
  $("#img-div").mousedown(function(e){
    is_down = true;
    drag_start_X = e.pageX;
    drag_start_Y = e.pageY;
  });

  $("#img-div").mousemove(function(e){
    mouse_move_counter = (mouse_move_counter + 1) % 100
    if(is_down){
      var deltaX = e.pageX - drag_start_X;
      var deltaY = e.pageY - drag_start_Y;
      if (mouse_move_counter % 10 == 0){
        adjust_position(deltaX, deltaY);
        drag_start_X = e.pageX;
        drag_start_Y = e.pageY;
      };
    }
  });

  $("#img-div").mouseup(function(e){
    is_down = false;
  });

  // drag control ends

  $("#zoom-in").on("click", function(){ zoomin() });
  $("#zoom-out").on("click", function(){ zoomout() });
});
